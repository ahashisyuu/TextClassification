# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
import pickle as pkl
from a02_TextCNN.p7_TextCNN_model import TextCNN
# from a02_TextCNN.data_util import create_vocabulary,load_data_multilabel
import os
# import word2vec

#configuration
FLAGS=tf.flags.FLAGS

tf.flags.DEFINE_string("traning_data_path", "../data/sample_multiple_label.txt","path of traning data.") #sample_multiple_label.txt-->train_label_single100_merge
tf.flags.DEFINE_integer("vocab_size", 323069, "maximum vocab size.")

tf.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 2000, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 10, "number of epochs to run.")
tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
# tf.flags.DEFINE_boolean("use_embedding", False, "whether to use embedding or not.")
tf.flags.DEFINE_integer("num_filters", 128, "number of filters")  # 256--->512
# tf.flags.DEFINE_string("word2vec_model_path", "word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
filter_sizes = [6, 7, 8]


def main(_):
    with open('../data_preprocessed/data_train_keras.pkl', 'rb') as fw_train, \
            open('../data_preprocessed/data_test_keras.pkl', 'rb') as fw_test:
        trainX, trainY = pkl.load(fw_train)
        testX = pkl.load(fw_test)
    # trainY = trainY.argmax(axis=-1)

    print("length of training data:", len(trainX), ";length of validation data:",len(testX))
    print("trainX[0]:", trainX[0])
    print("trainY[0]:", trainY[0])

    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, 19,
                          FLAGS.learning_rate, FLAGS.batch_size,
                          FLAGS.decay_steps, FLAGS.decay_rate, FLAGS.sentence_len,
                          323069,
                          FLAGS.embed_size, FLAGS.is_training,
                          multi_label_flag=FLAGS.multi_label_flag)
        # Initialize Save
        saver = tf.train.Saver(max_to_keep=10)
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
        curr_epoch = sess.run(textCNN.epoch_step)
        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        iteration = 0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter = 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration = iteration+1
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                feed_dict = {textCNN.input_x: trainX[start:end], textCNN.dropout_keep_prob: 0.5,textCNN.iter: iteration,textCNN.tst: not FLAGS.is_training}
                feed_dict[textCNN.input_y_multilabel]=trainY[start:end]
                curr_loss, lr, _, _ = sess.run([textCNN.loss_val,textCNN.learning_rate,textCNN.update_ema,textCNN.train_op],feed_dict)
                loss,counter = loss+curr_loss,counter+1
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            if epoch % FLAGS.validate_every == 0:
                # save model to checkpoint
                if os.path.exists(FLAGS.ckpt_dir.split('/')[0]) is False:
                    os.mkdir(FLAGS.ckpt_dir.split('/')[0])
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess,textCNN,evalX,evalY,iteration):
    number_examples=len(evalX)
    eval_loss,eval_counter,eval_f1_score,eval_p,eval_r=0.0,0,0.0,0.0,0.0
    batch_size=1
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.input_y_multilabel:evalY[start:end],textCNN.dropout_keep_prob: 1.0,textCNN.iter: iteration,textCNN.tst: True}
        curr_eval_loss, logits= sess.run([textCNN.loss_val,textCNN.logits],feed_dict)#curr_eval_acc--->textCNN.accuracy
        label_list_top5 = get_label_using_logits(logits[0])
        f1_score,p,r=compute_f1_score(list(label_list_top5), evalY[start:end][0])
        eval_loss,eval_counter,eval_f1_score,eval_p,eval_r=eval_loss+curr_eval_loss,eval_counter+1,eval_f1_score+f1_score,eval_p+p,eval_r+r
    return eval_loss/float(eval_counter),eval_f1_score/float(eval_counter),eval_p/float(eval_counter),eval_r/float(eval_counter)

def compute_f1_score(label_list_top5,eval_y):
    """
    compoute f1_score.
    :param logits: [batch_size,label_size]
    :param evalY: [batch_size,label_size]
    :return:
    """
    num_correct_label=0
    eval_y_short=get_target_label_short(eval_y)
    for label_predict in label_list_top5:
        if label_predict in eval_y_short:
            num_correct_label=num_correct_label+1
    #P@5=Precision@5
    num_labels_predicted=len(label_list_top5)
    all_real_labels=len(eval_y_short)
    p_5=num_correct_label/num_labels_predicted
    #R@5=Recall@5
    r_5=num_correct_label/all_real_labels
    f1_score=2.0*p_5*r_5/(p_5+r_5+0.000001)
    return f1_score,p_5,r_5

def get_target_label_short(eval_y):
    eval_y_short=[] #will be like:[22,642,1391]
    for index,label in enumerate(eval_y):
        if label>0:
            eval_y_short.append(index)
    return eval_y_short

#get top5 predicted labels
def get_label_using_logits(logits,top_number=5):
    index_list=np.argsort(logits)[-top_number:]
    index_list=index_list[::-1]
    return index_list

#统计预测的准确率
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

# def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,textCNN,word2vec_model_path):
#     print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
#     word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
#     word2vec_dict = {}
#     for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
#         word2vec_dict[word] = vector
#     word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
#     word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
#     bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
#     count_exist = 0;
#     count_not_exist = 0
#     for i in range(1, vocab_size):  # loop each word
#         word = vocabulary_index2word[i]  # get a word
#         embedding = None
#         try:
#             embedding = word2vec_dict[word]  # try to get vector:it is an array.
#         except Exception:
#             embedding = None
#         if embedding is not None:  # the 'word' exist a embedding
#             word_embedding_2dlist[i] = embedding;
#             count_exist = count_exist + 1  # assign array to this word.
#         else:  # no embedding for this word
#             word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
#             count_not_exist = count_not_exist + 1  # init a random value for the word.
#     word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
#     word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
#     t_assign_embedding = tf.assign(textCNN.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
#     sess.run(t_assign_embedding);
#     print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
#     print("using pre-trained word emebedding.ended...")


if __name__ == "__main__":
    tf.app.run()