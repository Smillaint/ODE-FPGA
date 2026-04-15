#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "_0.h"

int main(int argc, char* argv[])
{
    /* ===== 命令行参数解析 ===== */
    if (argc < 6) {
        printf("用法: selector.exe <grad_file> <global_grad_file>"
               " <round_num> <buffer_size> <speed>\n");
        return 1;
    }

    const char* grad_file_path   = argv[1];
    const char* global_grad_path = argv[2];
    int   ROUND    = atoi(argv[3]);
    int   buf_size = atoi(argv[4]);
    int   speed    = atoi(argv[5]);

    /* ===== 动态分配缓冲区 ===== */
    struct selected_buffer* buf = (struct selected_buffer*)malloc(
        buf_size * sizeof(struct selected_buffer));
    if (buf == NULL) {
        printf("内存分配失败\n");
        return 1;
    }
    for (int k = 0; k < buf_size; k++) {
        buf[k].index = -1;
        buf[k].val   = -1e4;
    }

    FILE*  file1, *file2;
    double grad[LENGTH];
    double global_grad[LENGTH];
    char   line[MAX_LINE_LENGTH];
    int    i, j, flag = 0, number;
    double val0 = 0.0;

    /* ===== 读取全局梯度 ===== */
    file2 = fopen(global_grad_path, "r");
    if (file2 == NULL) {
        printf("无法打开文件global_grad.txt\n");
        free(buf);
        return 1;
    }

    while (fgets(line, MAX_LINE_LENGTH, file2) != NULL) {
        if (sscanf(line, "Round %d", &number) == 1 && number == ROUND) {
            flag = 1;
            break;
        }
    }
    if (!flag) {
        printf("未找到Round %d\n", ROUND);
        fclose(file2);
        free(buf);
        return 1;
    }

    for (i = 0; i < LENGTH; i++) {
        if (fscanf(file2, "%lf,", &global_grad[i]) != 1) {
            printf("读取 global_grad 数据时出错\n");
            fclose(file2);
            free(buf);
            return 1;
        }
    }
    fclose(file2);

    /* ===== 读取客户端梯度文件 ===== */
    file1 = fopen(grad_file_path, "r");
    if (file1 == NULL) {
        printf("无法打开文件%s\n", grad_file_path);
        free(buf);
        return 1;
    }

    /* 找到对应轮次 */
    while (fgets(line, MAX_LINE_LENGTH, file1) != NULL)
        if (sscanf(line, "Round %d", &number) == 1 && number == ROUND)
            break;

    /* 跳过 "Initial:" 行 */
    fgets(line, MAX_LINE_LENGTH, file1);

    /* ===== 缓冲区初始化：读取前 buf_size 个样本 ===== */
    for (j = 0; j < buf_size; j++) {
        while (fgets(line, MAX_LINE_LENGTH, file1) != NULL)
            if (sscanf(line, "Data index %d", &number) == 1)
                break;

        for (i = 0; i < LENGTH; i++) {
            if (fscanf(file1, "%lf,", &grad[i]) != 1) {
                printf("读取 grad 数据时出错（初始化阶段）\n");
                fclose(file1);
                free(buf);
                return 1;
            }
        }
        val0 = calculate(grad, global_grad, buf, number, buf_size);
    }

    /* ===== 流式处理剩余样本，更新缓冲区 ===== */
    for (j = ROUND * speed + buf_size; j < (ROUND + 1) * speed; j++) {
        while (fgets(line, MAX_LINE_LENGTH, file1) != NULL)
            if (sscanf(line, "Data index %d", &number) == 1)
                break;

        for (i = 0; i < LENGTH; i++) {
            if (fscanf(file1, "%lf,", &grad[i]) != 1) {
                printf("读取 grad 数据时出错（流式阶段）\n");
                fclose(file1);
                free(buf);
                return 1;
            }
        }
        val0 = calculate(grad, global_grad, buf, number, buf_size);
    }
    fclose(file1);

    /* ===== 输出选中索引到 stdout（Python 解析）===== */
    for (i = 0; i < buf_size; i++) {
        if (buf[i].index >= 0)
            printf("Selected: %d\n", buf[i].index);
    }

    free(buf);
    return 0;
}
