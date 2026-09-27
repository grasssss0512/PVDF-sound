#ifndef ADC_APP_H
#define ADC_APP_H

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 默认采样率（每通道，Hz）
 *
 * 使用 ESP32-C3 内置 ADC1（12bit）。esp_timer 每 1/采样率 秒触发一次，
 * 每个触发点对启用通道各读 1 个样本，故每通道实际采样率 = 该值。
 */
#define ADC_SAMPLE_RATE_HZ          2000

/* 最大支持的采样通道数（环形缓冲区数组按此大小分配） */
#define ADC_CHANNEL_COUNT_MAX       2

/* 实际启用的采样通道数（<= ADC_CHANNEL_COUNT_MAX，修改后重新编译即可）
 * 本项目使用单通道：ADC1_CH1 / GPIO1 */
#define ADC_CHANNEL_COUNT           1

/* 单通道环形缓冲区大小（样本数），约为 采样率 * 2 秒 */
#define ADC_RING_BUF_SIZE           4096

/**
 * @brief 初始化 ESP32-C3 内置 ADC 采样（oneshot + esp_timer 定时采样）
 *
 * 通道与引脚映射见 adc_app.c 顶部的「ADC 采样参数配置区」。
 *
 * @param sample_rate_hz 每通道采样率（Hz），传 0 使用默认 ADC_SAMPLE_RATE_HZ
 * @return ESP_OK 成功，其他值失败
 */
esp_err_t adc_app_init(uint32_t sample_rate_hz);

/**
 * @brief 停止采样并释放所有资源
 */
void adc_app_deinit(void);

/**
 * @brief 获取实际采样率（Hz）
 */
uint32_t adc_app_get_sample_rate(void);

/**
 * @brief 获取启用的采样通道数
 */
int adc_app_get_channel_count(void);

/**
 * @brief 读取某通道的原始样本（环形缓冲区，FIFO 顺序）
 *
 * @param channel_idx 通道索引 0 ~ ADC_CHANNEL_COUNT-1
 * @param samples     输出缓冲区
 * @param max_count   最多读取的样本数
 * @return 实际返回的样本数
 */
int adc_app_read_samples(int channel_idx, int16_t *samples, int max_count);

/**
 * @brief 查询某通道当前缓冲区内可用样本数
 */
uint32_t adc_app_samples_available(int channel_idx);

/**
 * @brief 查询某通道累计成功采样数（用于统计实际采样率是否达标）
 */
uint32_t adc_app_get_total_samples(int channel_idx);

#ifdef __cplusplus
}
#endif

#endif
