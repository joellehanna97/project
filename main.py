import os
import time
import math
import tensorflow as tf
import numpy as np
import tensorlayer as tl

from ops import *
from datasets import *
from model import *
from config import config, log_config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))

def train():

    ## create folders to save result images and trained model

    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    checkpoint_input_g = "./donotexist.npz"
    checkpoint_input_d = "./donotexist.npz"

    ###====================== PRE-LOAD DATA ===========================###
    train_video_folders = '/media/saeed-lts5/Data-Saeed/SuperResolution/youtube8m-dataset/frames'
    train_vid_list = sorted(tl.files.load_folder_list(path=train_video_folders))

    #set up placeholders
    t_image = tf.placeholder(tf.float32, [1,82,82,6], name = 't_video_input_to_SRGAN_generator')
    t_target_image = tf.placeholder(tf.float32, [1, 82, 82, 3], name='t_target_image')

    #Sample generated frame from generator:
    net_g = SRGAN_g(t_image, is_train=True, reuse=False)

    # Evaluate discrimator on real triplets
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    #maybe add vgg (?)
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    ###========================== DEFINE TRAIN OPS ==========================###
    # Calculate GAN losses
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    #vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    # maybe add vgg_loss, ms_ssim_loss, and clipping_loss (?)

    #self.ms_ssim_loss = tf.reduce_mean(-tf.log(tf_ms_ssim(g_mean_added_clipped, sing_mean_added_clipped)))
    #self.clipping_loss = tf.reduce_mean(tf.square(g_mean_added - g_mean_added_clipped))

    # Combine losses into single functions for the discriminator and the generator
    d_loss = d_loss1 + d_loss2
    g_loss = mse_loss + g_gan_loss #vgg_loss +

    # Record relevant variables
    ## Tensorboard summaries
    summary_1 = tf.summary.scalar("loss_mse", mse_loss)
    summary_2 = tf.summary.scalar("loss_g_gan", g_gan_loss)
    summary_3 = tf.summary.scalar("loss_g", g_loss)
    summary_4 = tf.summary.scalar("loss_d1", d_loss1)
    summary_5 = tf.summary.scalar("loss_d2", d_loss2)
    summary_6 = tf.summary.scalar("loss_d", d_loss)


    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_input_g, network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_input_d, network=net_d)

    ###============================= TRAINING ===============================###
    train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[0] + '/frames/', regx='.*.png', printable=False))
    #train_lr_vid_img_list = sorted(tl.files.load_file_list(path=train_lr_vid_list[0] + '/frames/', regx='.*.png', printable=False))

    #train_target_vid_imgs = tl.vis.read_images([train_hr_vid_img_list[15],train_hr_vid_img_list[45],train_hr_vid_img_list[75],train_hr_vid_img_list[105]], path=train_hr_vid_list[0] + '/frames/', n_threads=32)
    train_target_vid_imgs = tl.vis.read_images([train_hr_vid_img_list[15]], path=train_hr_vid_list[0] + '/frames/', n_threads=32)
    indices_1 = [14,16]
    train_vid_img_list_s = [train_vid_img_list[i] for i in indices_1]
    train_vid_imgs = tl.vis.read_images(train_vid_img_list_s, path=train_lr_vid_list[0] + '/frames/', n_threads=32)
    """
    indices_2 = [44,46]
    indices_3 = [74,76]
    indices_4 = [104,106]

    train_vid_imgs = tl.vis.read_images(train_lr_vid_img_list[15-1:15+2]+
					   train_lr_vid_img_list[45-1:45+2]+
					   train_lr_vid_img_list[75-1:75+2]+
					   train_lr_vid_img_list[105-1:105+2], path=train_lr_vid_list[0] + '/frames/', n_threads=32)
    """
    ## use first `batch_size` of train set to have a quick test during training
    train_vid_imgs = tl.prepro.threading_data(train_vid_imgs, fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)
    train_target_vid_imgs = tl.prepro.threading_data(train_target_vid_imgs, fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False) #328 328
    train_vid_seqs = [np.concatenate([train_vid_imgs[0], train_vid_imgs[1]],2)]
    """
    train_lr_vid_seqs = np.stack([np.concatenate([train_lr_vid_imgs[0], train_lr_vid_imgs[1], train_lr_vid_imgs[2]], 2),
			np.concatenate([train_lr_vid_imgs[3], train_lr_vid_imgs[4], train_lr_vid_imgs[5]], 2),
			np.concatenate([train_lr_vid_imgs[6], train_lr_vid_imgs[7], train_lr_vid_imgs[8]], 2),
			np.concatenate([train_lr_vid_imgs[9], train_lr_vid_imgs[10], train_lr_vid_imgs[11]], 2)])
    """
    tl.vis.save_images(train_vid_imgs, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(train_vid_seqs, [ni, ni], save_dir_ginit + '/_train_sample_96_1.png')
    """
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,3:6], [ni, ni], save_dir_ginit + '/_train_sample_96_2.png')
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,6:9], [ni, ni], save_dir_ginit + '/_train_sample_96_3.png')
    """
    #this is for GAN
    tl.vis.save_images(train_vid_imgs, [ni, ni], save_dir_gan + '/_train_sample_384.png')
    tl.vis.save_images(train_vid_seqs, [ni, ni], save_dir_gan + '/_train_sample_96_1.png')
    """
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,3:6], [ni, ni], save_dir_gan + '/_train_sample_96_2.png')
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,6:9], [ni, ni], save_dir_gan + '/_train_sample_96_3.png')
    """

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))

    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    train_vid_list = train_vid_list[0:12] #5000
    #train_lr_vid_list = train_lr_vid_list[0:12] #5000
    for epoch in range(0, n_epoch_init + 1):

        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        #for idx in range(0, len(train_hr_imgs), batch_size):
        for idx in range(0, len(train_vid_list)):
            step_time = time.time()

            #b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            #b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            # add here the other images
            # b_imgs_96_2 = tl.prepro.threading_data()

            train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[idx] + '/frames/', regx='.*.png', printable=False))
            #train_lr_vid_img_list = sorted(tl.files.load_file_list(path=train_lr_vid_list[idx] + '/frames/', regx='.*.png', printable=False))

            #b_imgs_384 = tl.vis.read_images([train_hr_vid_img_list[15],
            #                                        train_hr_vid_img_list[45],train_hr_vid_img_list[75],train_hr_vid_img_list[105]], path=train_hr_vid_list[idx] + '/frames/', n_threads=32)
            b_imgs_384 = tl.vis.read_images([train_vid_img_list[15]], path=train_vid_list[idx] + '/frames/', n_threads=32) #target
            #b_imgs_96 = tl.vis.read_images(train_lr_vid_img_list[15-1:15+2]+
			#		           train_lr_vid_img_list[45-1:45+2]+
			#		           train_lr_vid_img_list[75-1:75+2]+
			#		           train_lr_vid_img_list[105-1:105+2], path=train_lr_vid_list[idx] + '/frames/', n_threads=32)
            train_vid_img_list_s = [train_vid_img_list[i] for i in indices_1]
            b_imgs_96 = tl.vis.read_images(train_vid_img_list_s, path=train_vid_list[idx] + '/frames/', n_threads=32)
            """
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=crop_custom, w=82, h=82, is_random=False)

            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=crop_custom, w=328, h=328, is_random=False)
            """
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)

            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)
            #b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=tl.prepro.crop, wrg=160, hrg=160, is_random=False)
            #b_seqs_96 = np.stack([np.concatenate([b_imgs_96[0], b_imgs_96[1], b_imgs_96[2]], 2),
			#        np.concatenate([b_imgs_96[3], b_imgs_96[4], b_imgs_96[5]], 2),
			#        np.concatenate([b_imgs_96[6], b_imgs_96[7], b_imgs_96[8]], 2),
			#        np.concatenate([b_imgs_96[9], b_imgs_96[10], b_imgs_96[11]], 2)])
            b_seqs_96 = [np.concatenate([b_imgs_96[0], b_imgs_96[1]],2)]

            ## update G
            #b_imgs_96_c = np.concatenate((b_imgs_96, b_imgs_96), axis=3)
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_seqs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set

        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: train_lr_vid_seqs})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

###========================= train GAN (SRGAN) =========================###
for epoch in range(0, n_epoch + 1):
    ## update learning rate
    if epoch != 0 and (epoch % decay_every == 0):
        new_lr_decay = lr_decay**(epoch // decay_every)
        sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        print(log)
    elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
        print(log)

    epoch_time = time.time()
    total_d_loss, total_g_loss, n_iter = 0, 0, 0

    ## If your machine cannot load all images into memory, you should use
    ## this one to load batch of images while training.
    # random.shuffle(train_hr_img_list)
    # for idx in range(0, len(train_hr_img_list), batch_size):
    #     step_time = time.time()
    #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
    #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
    #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
    #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

    # shuffle images
    step_time = time.time()
    train_all_img_list = list(zip(train_vid_list))
    random.shuffle(train_all_img_list)
    train_vid_list = zip(*train_all_img_list)
    print("Shuffled time: %4.4fs" % (time.time() - step_time))

    ## If your machine have enough memory, please pre-load the whole train set.
    for idx in range(0, len(train_vid_list)):
        step_time = time.time()
        """
        b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
        b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
        """
        train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[idx] + '/frames/', regx='.*.png', printable=False))
        #train_lr_vid_img_list = sorted(tl.files.load_file_list(path=train_lr_vid_list[idx] + '/frames/', regx='.*.png', printable=False))

        b_imgg_384 = tl.vis.read_images([train_vid_img_list[15]], path=train_vid_list[idx] + '/frames/', n_threads=32) #target
        #b_imgs_96 = tl.vis.read_images(train_lr_vid_img_list[15-1:15+2]+
        #		           train_lr_vid_img_list[45-1:45+2]+
        #		           train_lr_vid_img_list[75-1:75+2]+
        #		           train_lr_vid_img_list[105-1:105+2], path=train_lr_vid_list[idx] + '/frames/', n_threads=32)
        train_vid_img_list_s = [train_vid_img_list[i] for i in indices_1]
        b_imgs_96 = tl.vis.read_images(train_vid_img_list_s, path=train_vid_list[idx] + '/frames/', n_threads=32)
        """
        b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=crop_custom, w=82, h=82, is_random=False)

        b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=crop_custom, w=328, h=328, is_random=False)
        """
        b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)

        b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)
        #b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=tl.prepro.crop, wrg=160, hrg=160, is_random=False)
        #b_seqs_96 = np.stack([np.concatenate([b_imgs_96[0], b_imgs_96[1], b_imgs_96[2]], 2),
        #        np.concatenate([b_imgs_96[3], b_imgs_96[4], b_imgs_96[5]], 2),
        #        np.concatenate([b_imgs_96[6], b_imgs_96[7], b_imgs_96[8]], 2),
        #        np.concatenate([b_imgs_96[9], b_imgs_96[10], b_imgs_96[11]], 2)])
        b_seqs_96 = [np.concatenate([b_imgs_96[0], b_imgs_96[1]],2)]
        ## update D
        #b_imgs_96_c = np.concatenate((b_imgs_96, b_imgs_96), axis=3)
        errD, _ = sess.run([d_loss, d_optim], {t_image: b_seqs_96, t_target_image: b_imgs_384})
        ## update G
        errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim], {t_image: b_seqs_96, t_target_image: b_imgs_384})
        print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)" %
              (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
        total_d_loss += errD
        total_g_loss += errG
        n_iter += 1

    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                            total_g_loss / n_iter)
    print(log)
    """
    ## quick evaluation on train set
    if (epoch != 0) and (epoch % 10 == 0):
        out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
        print("[*] save images")
        tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

    ## save model
    if (epoch != 0) and (epoch % 10 == 0):
        tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
        tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
    """
