#ifndef _0_H
#define _0_H

#define MAX_LINE_LENGTH 1024   // 每行最大字符数
#define LENGTH          610    // 梯度数据长度

/* 缓冲区节点：优先队列 */
struct selected_buffer {
    int    index;   // 样本数据序号
    double val;     // 数据重要性
};

/* buf_size 由命令行传入，不再用宏 BUFFER */
double calculate(double grad[LENGTH],
                 double global_grad[LENGTH],
                 struct selected_buffer* buf,
                 int number,
                 int buf_size);

#endif
