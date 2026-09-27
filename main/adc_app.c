#include "adc_app.h"

#include <inttypes.h>

#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define TAG "ADC_APP"

/* ===================================================================
 *  ADC 采样参数配置区 - 用户可在此处调整参数
 * =================================================================== */

/*
 * ESP32-C3 内部 ADC1 可用通道与引脚对应关系：
 *
 *   ADC1_CH0 = GPIO0   ADC1_CH1 = GPIO1
 *   ADC1_CH2 = GPIO2   ADC1_CH3 = GPIO3
 *   ADC1_CH4 = GPIO4
 *
 * 引脚选择注意事项：
 *  - GPIO0 在最小系统板上一般接 BOOT 按键（上电时按下会进入下载模式），
 *    作为模拟输入可用但不推荐。
 *  - GPIO2 / GPIO8 / GPIO9 为 strapping 引脚，外接强下拉/强上拉电路会影响
 *    上电启动状态，请避开。
 *  - ADC2（GPIO5~GPIO9）与 Wi-Fi 共用硬件，Wi-Fi 开启时无法使用，
 *    故本项目只使用 ADC1。
 */

/* 第一路信号（如 ECG 心电）：接 GPIO1 */
#define SIG1_ADC_CHANNEL        ADC_CHANNEL_1
#define SIG1_ADC_GPIO           GPIO_NUM_1

/* 第二路信号（如 PCG 心音）：接 GPIO4 */
#define SIG2_ADC_CHANNEL        ADC_CHANNEL_4
#define SIG2_ADC_GPIO           GPIO_NUM_4

/* 量程：ADC_ATTEN_DB_12 = 0 ~ 3.1V（推荐，输入信号应偏置在约 1.5V 中点） */
#define ADC_ATTEN               ADC_ATTEN_DB_12
/* 分辨率：12bit，原始值范围 0 ~ 4095 */
#define ADC_BIT_WIDTH           ADC_BITWIDTH_12

/* =================================================================== */

static adc_oneshot_unit_handle_t s_adc_handle = NULL;
static esp_timer_handle_t s_sample_timer = NULL;

static uint32_t s_sample_rate_hz = ADC_SAMPLE_RATE_HZ;
static uint32_t s_sample_interval_us = 0;
static bool s_initialized = false;
static int s_chan_count = ADC_CHANNEL_COUNT;

/*
 * 单生产者（esp_timer 回调）单消费者（读取任务）的环形缓冲区。
 * 生产者只写 write_idx，消费者只写 read_idx，无需互斥锁。
 */
typedef struct {
    adc_channel_t channel;              /* ADC 通道号 */
    gpio_num_t    gpio;                 /* 对应 GPIO */
    int16_t       ring[ADC_RING_BUF_SIZE];
    volatile uint32_t write_idx;
    volatile uint32_t read_idx;
    volatile uint32_t total;            /* 累计成功采样数 */
} adc_channel_ctx_t;

static adc_channel_ctx_t s_chan[ADC_CHANNEL_COUNT_MAX] = {
    { .channel = SIG1_ADC_CHANNEL, .gpio = SIG1_ADC_GPIO },
    { .channel = SIG2_ADC_CHANNEL, .gpio = SIG2_ADC_GPIO },
};

static void adc_ring_push(adc_channel_ctx_t *c, int16_t val)
{
    c->ring[c->write_idx] = val;
    c->write_idx = (c->write_idx + 1) % ADC_RING_BUF_SIZE;
    /* 写指针追上读指针时丢弃最旧样本 */
    if (c->write_idx == c->read_idx) {
        c->read_idx = (c->read_idx + 1) % ADC_RING_BUF_SIZE;
    }
    c->total++;
}

static int adc_ring_pop(adc_channel_ctx_t *c, int16_t *out, int max)
{
    int n = 0;
    while (n < max && c->read_idx != c->write_idx) {
        out[n++] = c->ring[c->read_idx];
        c->read_idx = (c->read_idx + 1) % ADC_RING_BUF_SIZE;
    }
    return n;
}

/**
 * @brief esp_timer 定时采样回调
 *
 * 每 s_sample_interval_us 微秒触发一次，对所有启用通道各读 1 个样本并写入环形缓冲。
 * 在 esp_timer 任务上下文中执行，仅做短操作，不阻塞。
 */
static void sample_timer_cb(void *arg)
{
    (void)arg;

    for (int i = 0; i < s_chan_count; i++) {
        int raw = -1;
        if (adc_oneshot_read(s_adc_handle, s_chan[i].channel, &raw) == ESP_OK && raw >= 0) {
            adc_ring_push(&s_chan[i], (int16_t)raw);
        }
    }
}

esp_err_t adc_app_init(uint32_t sample_rate_hz)
{
    esp_err_t ret;

    if (s_initialized) {
        return ESP_OK;
    }

    if (sample_rate_hz == 0) {
        sample_rate_hz = ADC_SAMPLE_RATE_HZ;
    }
    s_sample_rate_hz = sample_rate_hz;
    s_sample_interval_us = 1000000 / sample_rate_hz;

    if (s_chan_count > ADC_CHANNEL_COUNT_MAX) {
        s_chan_count = ADC_CHANNEL_COUNT_MAX;
    }

    /* 创建 oneshot ADC 单元（ADC1，与 Wi-Fi 无冲突） */
    const adc_oneshot_unit_init_cfg_t unit_cfg = {
        .unit_id = ADC_UNIT_1,
        .clk_src = ADC_DIGI_CLK_SRC_DEFAULT,
    };
    ret = adc_oneshot_new_unit(&unit_cfg, &s_adc_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "adc_oneshot_new_unit failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* 配置各通道 */
    const adc_oneshot_chan_cfg_t chan_cfg = {
        .atten = ADC_ATTEN,
        .bitwidth = ADC_BIT_WIDTH,
    };
    for (int i = 0; i < s_chan_count; i++) {
        /* 先复位引脚（去除默认上拉/下拉），再交给 ADC 外设 */
        gpio_reset_pin(s_chan[i].gpio);
        ret = adc_oneshot_config_channel(s_adc_handle, s_chan[i].channel, &chan_cfg);
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "adc_oneshot_config_channel(CH%d) failed: %s",
                     i, esp_err_to_name(ret));
            adc_oneshot_del_unit(s_adc_handle);
            s_adc_handle = NULL;
            return ret;
        }
        ESP_LOGI(TAG, "Channel %d: ADC1_CH%d @ GPIO%d",
                 i, (int)s_chan[i].channel, (int)s_chan[i].gpio);
    }

    /* 创建采样定时器 */
    const esp_timer_create_args_t timer_args = {
        .callback = sample_timer_cb,
        .name = "adc_timer",
    };
    ret = esp_timer_create(&timer_args, &s_sample_timer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_timer_create failed: %s", esp_err_to_name(ret));
        adc_oneshot_del_unit(s_adc_handle);
        s_adc_handle = NULL;
        return ret;
    }

    ret = esp_timer_start_periodic(s_sample_timer, s_sample_interval_us);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_timer_start_periodic failed: %s", esp_err_to_name(ret));
        esp_timer_delete(s_sample_timer);
        s_sample_timer = NULL;
        adc_oneshot_del_unit(s_adc_handle);
        s_adc_handle = NULL;
        return ret;
    }

    s_initialized = true;

    ESP_LOGI(TAG, "ADC initialized: %d channel(s) @ %" PRIu32 " Hz (%d us/tick)",
             s_chan_count, s_sample_rate_hz, (int)s_sample_interval_us);
    return ESP_OK;
}

void adc_app_deinit(void)
{
    if (!s_initialized) {
        return;
    }

    if (s_sample_timer) {
        esp_timer_stop(s_sample_timer);
        esp_timer_delete(s_sample_timer);
        s_sample_timer = NULL;
    }

    if (s_adc_handle) {
        adc_oneshot_del_unit(s_adc_handle);
        s_adc_handle = NULL;
    }

    for (int i = 0; i < ADC_CHANNEL_COUNT_MAX; i++) {
        s_chan[i].write_idx = 0;
        s_chan[i].read_idx = 0;
        s_chan[i].total = 0;
    }
    s_initialized = false;

    ESP_LOGI(TAG, "ADC deinitialized");
}

uint32_t adc_app_get_sample_rate(void)
{
    return s_sample_rate_hz;
}

int adc_app_get_channel_count(void)
{
    return s_chan_count;
}

int adc_app_read_samples(int channel_idx, int16_t *samples, int max_count)
{
    if (channel_idx < 0 || channel_idx >= s_chan_count || samples == NULL || max_count <= 0) {
        return 0;
    }
    return adc_ring_pop(&s_chan[channel_idx], samples, max_count);
}

uint32_t adc_app_samples_available(int channel_idx)
{
    if (channel_idx < 0 || channel_idx >= s_chan_count) return 0;
    adc_channel_ctx_t *c = &s_chan[channel_idx];
    return (c->write_idx - c->read_idx + ADC_RING_BUF_SIZE) % ADC_RING_BUF_SIZE;
}

uint32_t adc_app_get_total_samples(int channel_idx)
{
    if (channel_idx < 0 || channel_idx >= s_chan_count) return 0;
    return s_chan[channel_idx].total;
}
