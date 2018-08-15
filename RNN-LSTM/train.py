"""

@Author  : dilless
@Time    : 2018/8/14 20:59
@File    : train.py
"""
import datetime

import numpy as np

from network import *
from utils import *


def train(train_data, vocab_size, num_layers, num_epochs, batch_size, model_save_name, learning_rate=1.0,
          max_lr_epoch=10, lr_decay=0.93, print_iter=50):
    """

    :param train_data:
    :param vocab_size:
    :param num_layers:
    :param num_epochs:
    :param batch_size:
    :param model_save_name:
    :param learning_rate:
    :param max_lr_epoch:
    :param lr_decay:
    :param print_iter:
    :return:
    """
    # 训练的输入
    training_input = Input(batch_size=batch_size, num_steps=35, data=train_data)

    # 创建训练的模型
    m = Model(training_input, is_training=True, hidden_size=650, vocab_size=vocab_size, num_layers=num_layers)

    # 初始化变量
    init_op = tf.global_variables_initializer()

    # 初始的学习率的衰减率
    orig_decay = lr_decay

    with tf.Session() as sess:
        sess.run(init_op)  # 初始化所有变量

        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 为了用 Saver 来保存模型的变量
        saver = tf.train.Saver()  # max_to_keep 默认是 5, 只保存最近的 5 个模型参数文件

        # 开始 Epoch 的训练
        for epoch in range(num_epochs):
            # 只有 Epoch 数大于 max_lr_epoch（设置为 10）后，才会使学习率衰减
            # 也就是说前 10 个 Epoch 的学习率一直是 1, 之后每个 Epoch 学习率都会衰减
            new_lr_decay = orig_decay ** max(epoch + 1 - max_lr_epoch, 0)
            m.assign_lr(sess, learning_rate * new_lr_decay)

            # 当前的状态
            # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
            # 一个是 前一时刻 LSTM 的输出 h(t-1)
            # 一个是 前一时刻的单元状态 C(t-1)
            current_state = np.zeros((num_layers, 2, batch_size, m.hidden_size))

            # 获取当前时间，以便打印日志时用
            current_time = datetime.datetime.now()

            for step in range(training_input.epoch_size):
                # train_op 操作：计算被修剪（clipping）过的梯度，并最小化 cost（误差）
                # state 操作：返回时间维度上展开的最后 LSTM 单元的输出（C(t) 和 h(t)），作为下一个 Batch 的输入状态
                if step % print_iter != 0:
                    cost, _, current_state = sess.run([m.cost, m.train_op, m.state],
                                                      feed_dict={m.init_state: current_state})
                else:
                    seconds = float((datetime.datetime.now() - current_time).seconds) / print_iter
                    current_time = datetime.datetime.now()
                    cost, _, current_state, acc = sess.run([m.cost, m.train_op, m.state, m.accuracy],
                                                           feed_dict={m.init_state: current_state})
                    print('Epoch {}, 第 {} 步, 损失: {:.3f}, 精度: {:.3f}, 每步所用时间: {:.3f}s'.format(epoch, step, cost, acc,
                                                                                             seconds))

            # 保存一个模型的变量的 checkpoint 文件
            saver.save(sess, save_path + '/' + model_save_name, global_step=epoch)

        # 对模型做一次总的保存
        saver.save(sess, save_path + '/' + model_save_name + '-final')

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ in ['builtins', '__main__']:
    if args.data_path:
        data_path = args.data_path
    if args.save_path:
        save_path = args.save_path
    print('data_path: ' + data_path)
    print('save_path: ' + save_path)
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)
    train(train_data, vocab_size, num_layers=2, num_epochs=50, batch_size=100,
          model_save_name='train-checkpoint')
