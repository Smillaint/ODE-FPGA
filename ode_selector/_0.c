#include "_0.h"

double calculate(double grad[LENGTH],
                 double global_grad[LENGTH],
                 struct selected_buffer* buf,
                 int number,
                 int buf_size)
{
    int i, k;
    double val = 0.0;

    /* 计算全局梯度与样本梯度的内积 */
    for (i = 0; i < LENGTH; i++)
        val += grad[i] * global_grad[i];

    /* 确定在优先队列中的插入位置 */
    for (i = 0; i < buf_size; i++)
        if (val < buf[i].val)
            break;

    if (i != 0) {
        k = i - 1;
        /* 向左移位，腾出位置 */
        for (i = 0; i < k; i++)
            buf[i] = buf[i + 1];
        buf[k].val   = val;
        buf[k].index = number;
    }

    return val;
}
