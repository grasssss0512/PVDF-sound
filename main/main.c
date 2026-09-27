#include <stdio.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_err.h"

#include "adc_app.h"

#define TAG "MAIN"

/* 每行最多样本数（保持单行 < 1024 字节） */
#define STREAM_BATCH_SIZE   200
/* 健康日志间隔（样本数 = 采样率 × 秒数） */
#define HEALTH_LOG_PERIOD   (10 * ADC_SAMPLE_RATE_HZ)

/**
 * @brief 将一批样本以 D:v1,v2,...,vN\n 格式输出到控制台
 *
 * ESP-IDF 控制台 vfs 会把 stdout 同时写入主串口(UART0)和副串口(USB Serial/JTAG)，
 * 因此上位机通过 USB(COM 口) 读取到该数据。上位机只解析以 'D:' 开头的行，其余为日志。
 */
static void stream_print_batch(const int16_t *samples, int n)
{
    char buf[1024];
    int pos = 0;

    buf[pos++] = 'D';
    buf[pos++] = ':';
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            buf[pos++] = ',';
        }
        /* 手动 uint16 -> 十进制，比 sprintf 快 */
        unsigned int v = (unsigned int)samples[i];
        if (v >= 1000) buf[pos++] = '0' + (v / 1000) % 10;
        if (v >= 100)  buf[pos++] = '0' + (v / 100) % 10;
        if (v >= 10)   buf[pos++] = '0' + (v / 10) % 10;
        buf[pos++] = '0' + v % 10;
    }
    buf[pos++] = '\n';

    fwrite(buf, 1, (size_t)pos, stdout);
    fflush(stdout);
}

void app_main(void)
{
    /* 启动单通道 ADC 采样（采样率传入 0 表示使用默认值 ADC_SAMPLE_RATE_HZ） */
    esp_err_t ret = adc_app_init(ADC_SAMPLE_RATE_HZ);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "ADC init failed: %s", esp_err_to_name(ret));
        return;
    }
    ESP_LOGI(TAG, "ADC streaming started: %u Hz, output format D:v1,v2,...",
             (unsigned int)adc_app_get_sample_rate());

    int16_t buf[STREAM_BATCH_SIZE];
    uint32_t total = 0;
    uint32_t last_log_total = 0;

    while (1) {
        /* 读取环形缓冲中的样本并逐批发送到上位机 */
        int n = adc_app_read_samples(0, buf, STREAM_BATCH_SIZE);
        if (n > 0) {
            stream_print_batch(buf, n);
            total += (uint32_t)n;

            /* 每累计 10 秒样本数打印一次健康日志（上位机可忽略非 D: 行） */
            if (total - last_log_total >= HEALTH_LOG_PERIOD) {
                ESP_LOGI(TAG, "streamed %lu samples", (unsigned long)total);
                last_log_total = total;
            }
        } else {
            /* 缓冲为空时让出 CPU */
            vTaskDelay(pdMS_TO_TICKS(2));
        }
    }
}
