import os, time, pickle, random, time
from datetime import datetime
from time import localtime, strftime
import logging, scipy
import math
import tensorflow as tf
import numpy as np
import tensorlayer as tl
from utils import *
from scipy.misc import imsave
import re

from ops import *
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

train_only_generator = False
train_using_gan = True

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
    ###========================== DEFINE MODEL ============================###

    #set up placeholders
    #t_image = tf.placeholder(tf.float32, [1,82,82,6], name = 't_video_input_to_SRGAN_generator')
    #t_target_image = tf.placeholder(tf.float32, [1, 82, 82, 3], name='t_target_image')
    t_image = tf.placeholder(tf.float32, [4,82,82,6], name = 't_video_input_to_SRGAN_generator')

    t_target_image = tf.placeholder(tf.float32, [4, 82, 82, 3], name='t_target_image')

    # For the discrimator
    t_target_images_3 = tf.placeholder(tf.float32, [4, 82, 82, 9], name='t_target_images_3')
    t_images_3 = tf.placeholder(tf.float32, [4, 82, 82, 9], name='t_images_3')



    #Sample generated frame from generator:
    net_g = SRGAN_g(t_image, is_train=True, reuse=False)
    print(net_g.outputs.get_shape())


    # Evaluate discrimator on real triplets

    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    """
    # Evaluate discrimator on real triplets
    net_d, logits_real = SRGAN_d(t_target_images_3, is_train=True, reuse=False)
    # Evaluate discrimator on fake triplets
    _, logits_fake = SRGAN_d(t_images_3, is_train=True, reuse=True)

    """
    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)

    ssim_valid = tf.reduce_mean(tf.image.ssim(net_g_test.outputs, t_target_image, 1))
    mse_valid = tf.losses.mean_squared_error(net_g_test.outputs, t_target_image)
    #mse_valid = tf.losses.absolute_difference(net_g_test.outputs, t_target_image) # --> l1 loss



    ###========================== DEFINE TRAIN OPS ==========================###
    # Calculate GAN losses
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    #
    #mse_loss = tl.cost.absolute_difference_error(net_g.outputs, t_target_image, is_mean=True) # --> L1 loss
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)
    #vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    # maybe add vgg_loss, ms_ssim_loss, and clipping_loss (?)

    #self.ms_ssim_loss = tf.reduce_mean(-tf.log(tf_ms_ssim(g_mean_added_clipped, sing_mean_added_clipped)))
    #self.clipping_loss = tf.reduce_mean(tf.square(g_mean_added - g_mean_added_clipped))

    # Combine losses into single functions for the discriminator and the generator

    d_loss = d_loss1 + d_loss2
    g_loss = mse_loss + g_gan_loss + vgg_loss
    g_loss_2 = mse_loss + vgg_loss

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

    #tl.files.load_and_assign_npz(sess=sess, name= checkpoint_dir + '/g_srgan.npz', network=net_g)
    #tl.files.load_and_assign_npz(sess=sess, name= checkpoint_dir + '/d_srgan.npz', network=net_d)

    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_input_g, network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_input_d, network=net_d)
    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[0] + '/frames/', regx='.*.png', printable=False))
    #train_lr_vid_img_list = sorted(tl.files.load_file_list(path=train_lr_vid_list[0] + '/frames/', regx='.*.png', printable=False))

    #train_target_vid_imgs = tl.vis.read_images([train_hr_vid_img_list[15],train_hr_vid_img_list[45],train_hr_vid_img_list[75],train_hr_vid_img_list[105]], path=train_hr_vid_list[0] + '/frames/', n_threads=32)
    train_target_vid_imgs = tl.vis.read_images([train_vid_img_list[20],train_vid_img_list[50],train_vid_img_list[80],train_vid_img_list[110]], path=train_vid_list[0] + '/frames/', n_threads=32)
    #train_target_vid_imgs = tl.vis.read_images([train_vid_img_list[45]], path=train_vid_list[0] + '/frames/', n_threads=32)
    #indices_1 = [14,16]
    #indices_2 = [44,46]
    #indices_3 = [74,76]
    #indices_4 = [104,106]
    indices_1 = [19,21]
    indices_2 = [49,51]
    indices_3 = [79,81]
    indices_4 = [109,111]
    train_vid_img_list_s1 = [train_vid_img_list[i] for i in indices_1]
    train_vid_img_list_s2 = [train_vid_img_list[i] for i in indices_2]
    train_vid_img_list_s3 = [train_vid_img_list[i] for i in indices_3]
    train_vid_img_list_s4 = [train_vid_img_list[i] for i in indices_4]

    train_vid_imgs = tl.vis.read_images(train_vid_img_list_s1+train_vid_img_list_s2+train_vid_img_list_s3+train_vid_img_list_s4, path=train_vid_list[0] + '/frames/', n_threads=32)
    #train_vid_imgs = tl.vis.read_images(train_vid_img_list_s2, path=train_vid_list[0] + '/frames/', n_threads=32)


    """
    [train_hr_vid_img_list[15],
    train_hr_vid_img_list[45],train_hr_vid_img_list[75],train_hr_vid_img_list[105]]

    train_vid_imgs = tl.vis.read_images(train_lr_vid_img_list[15-1:15+2]+
					   train_lr_vid_img_list[45-1:45+2]+
					   train_lr_vid_img_list[75-1:75+2]+
					   train_lr_vid_img_list[105-1:105+2], path=train_lr_vid_list[0] + '/frames/', n_threads=32)
    """
    ## use first `batch_size` of train set to have a quick test during training
    train_vid_imgs = tl.prepro.threading_data(train_vid_imgs, fn = crop_sub_imgs_fn,is_random=False) #fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)
    train_target_vid_imgs = tl.prepro.threading_data(train_target_vid_imgs, fn = crop_sub_imgs_fn,is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False) #328 328
    #train_vid_seqs = [np.concatenate([train_vid_imgs[0], train_vid_imgs[1]],2)]

    train_vid_seqs = np.stack([np.concatenate([train_vid_imgs[0], train_vid_imgs[1]], 2),
			np.concatenate([train_vid_imgs[2], train_vid_imgs[3]], 2),
			np.concatenate([train_vid_imgs[4], train_vid_imgs[5]], 2),
			np.concatenate([train_vid_imgs[6], train_vid_imgs[7]], 2)])

    tl.vis.save_images(train_target_vid_imgs, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    #tl.vis.save_images(np.asarray(train_vid_seqs), [ni, ni], save_dir_ginit + '/_train_sample_96_1.png')
    tl.vis.save_images(train_vid_seqs[:,:,:,0:3], [ni, ni], save_dir_ginit + '/_train_sample_96_1.png')
    tl.vis.save_images(train_vid_seqs[:,:,:,3:6], [ni, ni], save_dir_ginit + '/_train_sample_96_2.png')

    """
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,3:6], [ni, ni], save_dir_ginit + '/_train_sample_96_2.png')
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,6:9], [ni, ni], save_dir_ginit + '/_train_sample_96_3.png')
    """
    #this is for GAN
    tl.vis.save_images(train_vid_seqs[:,:,:,0:3], [ni, ni], save_dir_gan + '/_train_sample_96_1.png')
    tl.vis.save_images(train_vid_seqs[:,:,:,3:6], [ni, ni], save_dir_gan + '/_train_sample_96_2.png')
    #tl.vis.save_images(train_vid_seqs, [ni, ni], save_dir_gan + '/_train_sample_96_1.png')
    """
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,3:6], [ni, ni], save_dir_gan + '/_train_sample_96_2.png')
    tl.vis.save_images(train_lr_vid_seqs[:,:,:,6:9], [ni, ni], save_dir_gan + '/_train_sample_96_3.png')
    """

    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))

    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    train_vid_list = train_vid_list[0:500] #5000

    for epoch in range(0, n_epoch_init + 1): #0

        if (train_only_generator == False):
            print("Training only generator is Off. Continuing to GAN ...")
            break

        ## Evaluation on train set (first 4 images of training set)
        if (epoch % 1 == 0):
            #start_time = time.time()
            #start_time = time.time()
            out = sess.run(net_g_test.outputs, {t_image: train_vid_seqs})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            #print("took: %4.4fs" % (time.time() - start_time))
            print("[Evauation] save training images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

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

            #b_imgs_384 = tl.vis.read_images([train_vid_img_list[15],
            #                                        train_vid_img_list[45],train_vid_img_list[75],train_vid_img_list[105]], path=train_vid_list[idx] + '/frames/', n_threads=32)
            b_imgs_384 = tl.vis.read_images([train_vid_img_list[20],train_vid_img_list[50],train_vid_img_list[80],train_vid_img_list[110]], path=train_vid_list[idx] + '/frames/', n_threads=32)
            #train_target_vid_imgs = tl.vis.read_images([train_vid_img_list[45]], path=train_vid_list[0] + '/frames/', n_threads=32)
            #indices_1 = [14,16]
            #indices_2 = [44,46]
            #indices_3 = [74,76]
            #indices_4 = [104,106]
            indices_1 = [19,21]
            indices_2 = [49,51]
            indices_3 = [79,81]
            indices_4 = [109,111]
            #b_imgs_384 = tl.vis.read_images([train_vid_img_list[45]], path=train_vid_list[idx] + '/frames/', n_threads=32) #target
            #b_imgs_96 = tl.vis.read_images(train_lr_vid_img_list[15-1:15+2]+
			#		           train_lr_vid_img_list[45-1:45+2]+
			#		           train_lr_vid_img_list[75-1:75+2]+
			#		           train_lr_vid_img_list[105-1:105+2], path=train_lr_vid_list[idx] + '/frames/', n_threads=32)
            train_vid_img_list_s1 = [train_vid_img_list[i] for i in indices_1]
            train_vid_img_list_s2 = [train_vid_img_list[i] for i in indices_2]
            train_vid_img_list_s3 = [train_vid_img_list[i] for i in indices_3]
            train_vid_img_list_s4 = [train_vid_img_list[i] for i in indices_4]
            b_imgs_96 = tl.vis.read_images(train_vid_img_list_s1+train_vid_img_list_s2+train_vid_img_list_s3+train_vid_img_list_s4, path=train_vid_list[idx] + '/frames/', n_threads=32)
            #b_imgs_96 = tl.vis.read_images(train_vid_img_list_s, path=train_vid_list[idx] + '/frames/', n_threads=32)
            """
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=crop_custom, w=82, h=82, is_random=False)

            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=crop_custom, w=328, h=328, is_random=False)
            """
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn = crop_sub_imgs_fn,is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)

            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn = crop_sub_imgs_fn,is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)
            #b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=tl.prepro.crop, wrg=160, hrg=160, is_random=False)
            #b_seqs_96 = np.stack([np.concatenate([b_imgs_96[0], b_imgs_96[1], b_imgs_96[2]], 2),
			#        np.concatenate([b_imgs_96[3], b_imgs_96[4], b_imgs_96[5]], 2),
			#        np.concatenate([b_imgs_96[6], b_imgs_96[7], b_imgs_96[8]], 2),
			#        np.concatenate([b_imgs_96[9], b_imgs_96[10], b_imgs_96[11]], 2)])
            #b_seqs_96 = [np.concatenate([b_imgs_96[0], b_imgs_96[1]],2)]
            b_seqs_96 = np.stack([np.concatenate([b_imgs_96[0], b_imgs_96[1]], 2),
        			np.concatenate([b_imgs_96[2], b_imgs_96[3]], 2),
        			np.concatenate([b_imgs_96[4], b_imgs_96[5]], 2),
        			np.concatenate([b_imgs_96[6], b_imgs_96[7]], 2)])

            ## update G
            #b_imgs_96_c = np.concatenate((b_imgs_96, b_imgs_96), axis=3)
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_seqs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)

        ## quick evaluation on train set

        if (epoch != 0) and (epoch % 1 == 0):
            out = sess.run(net_g_test.outputs, {t_image: train_vid_seqs})  #net_g_test.outputs#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
            #tl.vis.save_image(out, save_dir_ginit + '/train_%d.png' % epoch)
            print("successful")

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1): #94
        if (train_using_gan == False):
            print("Using GAN is deactivated. Exiting loop ...")
            break

        ## Evaluation on train set (first 4 images of training set)
        if (epoch % 1 == 0):
            out = sess.run(net_g_test.outputs, {t_image: train_vid_seqs  })
            print("[Evauation] save training images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)


        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            #sess.run(tf.assign(lr_v, lr_init))
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
        """
        step_time = time.time()
        train_all_img_list = list(zip(train_vid_list))
        random.shuffle(train_all_img_list)
        train_vid_list = zip(*train_all_img_list)

        print(type(train_vid_list))
        print("Shuffled time: %4.4fs" % (time.time() - step_time))
        """
        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_vid_list)):
            step_time = time.time()
            """
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            """
            train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[idx] + '/frames/', regx='.*.png', printable=False))
            #train_lr_vid_img_list = sorted(tl.files.load_file_list(path=train_lr_vid_list[idx] + '/frames/', regx='.*.png', printable=False))

            #b_imgs_384 = tl.vis.read_images([train_vid_img_list[45]], path=train_vid_list[idx] + '/frames/', n_threads=32) #target
            #b_imgs_384 = tl.vis.read_images([train_vid_img_list[15],
            #                            train_vid_img_list[45],train_vid_img_list[75],train_vid_img_list[105]], path=train_vid_list[idx] + '/frames/', n_threads=32)

            b_imgs_384 = tl.vis.read_images([train_vid_img_list[20],train_vid_img_list[50],train_vid_img_list[80],train_vid_img_list[110]], path=train_vid_list[idx] + '/frames/', n_threads=32)
            b_imgs_384_3 = tl.vis.read_images(train_vid_img_list[20-1:20+2]+
					           train_vid_img_list[50-1:50+2]+
					           train_vid_img_list[80-1:80+2]+
					           train_vid_img_list[110-1:110+2], path=train_vid_list[idx] + '/frames/', n_threads=32)
            #train_target_vid_imgs = tl.vis.read_images([train_vid_img_list[45]], path=train_vid_list[0] + '/frames/', n_threads=32)
            #indices_1 = [14,16]
            #indices_2 = [44,46]
            #indices_3 = [74,76]
            #indices_4 = [104,106]
            indices_1 = [19,21]
            indices_2 = [49,51]
            indices_3 = [79,81]
            indices_4 = [109,111]
            #b_imgs_96 = tl.vis.read_images(train_lr_vid_img_list[15-1:15+2]+
            #		           train_lr_vid_img_list[45-1:45+2]+
            #		           train_lr_vid_img_list[75-1:75+2]+
            #		           train_lr_vid_img_list[105-1:105+2], path=train_lr_vid_list[idx] + '/frames/', n_threads=32)
            #train_vid_img_list_s = [train_vid_img_list[i] for i in indices_2]
            #b_imgs_96 = tl.vis.read_images(train_vid_img_list_s, path=train_vid_list[idx] + '/frames/', n_threads=32)
            train_vid_img_list_s1 = [train_vid_img_list[i] for i in indices_1]
            train_vid_img_list_s2 = [train_vid_img_list[i] for i in indices_2]
            train_vid_img_list_s3 = [train_vid_img_list[i] for i in indices_3]
            train_vid_img_list_s4 = [train_vid_img_list[i] for i in indices_4]
            b_imgs_96 = tl.vis.read_images(train_vid_img_list_s1+train_vid_img_list_s2+train_vid_img_list_s3+train_vid_img_list_s4, path=train_vid_list[idx] + '/frames/', n_threads=32)
            """
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn=crop_custom, w=82, h=82, is_random=False)

            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=crop_custom, w=328, h=328, is_random=False)
            """
            b_imgs_96 = tl.prepro.threading_data(b_imgs_96, fn = crop_sub_imgs_fn,is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)

            b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn = crop_sub_imgs_fn,is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)#fn=tl.prepro.crop, wrg=82, hrg=82, is_random=False)
            b_imgs_384_3 = tl.prepro.threading_data(b_imgs_384_3, fn = crop_sub_imgs_fn,is_random=False)
            #b_imgs_384 = tl.prepro.threading_data(b_imgs_384, fn=tl.prepro.crop, wrg=160, hrg=160, is_random=False)
            #b_seqs_96 = np.stack([np.concatenate([b_imgs_96[0], b_imgs_96[1], b_imgs_96[2]], 2),
            #        np.concatenate([b_imgs_96[3], b_imgs_96[4], b_imgs_96[5]], 2),
            #        np.concatenate([b_imgs_96[6], b_imgs_96[7], b_imgs_96[8]], 2),
            #        np.concatenate([b_imgs_96[9], b_imgs_96[10], b_imgs_96[11]], 2)])
            #b_seqs_96 = [np.concatenate([b_imgs_96[0], b_imgs_96[1]],2)]
            b_seqs_96 = np.stack([np.concatenate([b_imgs_96[0], b_imgs_96[1]], 2),
        			np.concatenate([b_imgs_96[2], b_imgs_96[3]], 2),
        			np.concatenate([b_imgs_96[4], b_imgs_96[5]], 2),
        			np.concatenate([b_imgs_96[6], b_imgs_96[7]], 2)])
            b_seqs_384 = np.stack([np.concatenate([b_imgs_384_3[0], b_imgs_384_3[1],b_imgs_384_3[2] ], 2),
        			np.concatenate([b_imgs_384_3[3], b_imgs_384_3[4],b_imgs_384_3[5] ], 2),
        			np.concatenate([b_imgs_384_3[6], b_imgs_384_3[7],b_imgs_384_3[8] ], 2),
        			np.concatenate([b_imgs_384_3[9], b_imgs_384_3[10],b_imgs_384_3[11] ], 2)])

            """
            b_fake_3 = [train_vid_img_list[19], net_g.outputs[0], train_vid_img_list[21]]
            net_g.outputs
            print(np.shape(b_fake_3))
            """
            """
            b_fake_3 = np.stack([np.concatenate([b_imgs_384_3[0], net_g.outputs[0],b_imgs_384_3[2] ], 2),
        			np.concatenate([b_imgs_384_3[3], net_g.outputs[1],b_imgs_384_3[5] ], 2),
        			np.concatenate([b_imgs_384_3[6], net_g.outputs[2],b_imgs_384_3[8] ], 2),
        			np.concatenate([b_imgs_384_3[9], net_g.outputs[3],b_imgs_384_3[11] ], 2)])
            """

            #b_fake_3 = np.concatenate([b_imgs_384_3[0], net_g.outputs[0],b_imgs_384_3[2] ], 2)
            print('lala')
            print(type(b_imgs_384_3[0]))
            print(np.shape(b_imgs_384_3[0]))
            print(net_g.outputs[0].get_shape())
            print(type(np.asarray(net_g.outputs[0])))


            print(np.shape(b_imgs_384_3[2]))
            b_fake_3 = np.concatenate([b_imgs_384_3[0], b_imgs_384_3[2] ], 2)
            print('shapes')

            print(np.shape(b_seqs_384))
            print(np.shape(b_fake_3))
            ## update D
            #b_imgs_96_c = np.concatenate((b_imgs_96, b_imgs_96), axis=3)
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_seqs_96, t_target_image: b_imgs_384})
            ## update G
            errG, errM, errA, errV, _ = sess.run([g_loss, mse_loss, g_gan_loss,vgg_loss, g_optim], {t_image: b_seqs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV,  errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            out = sess.run(net_g_test.outputs, {t_image: train_vid_seqs})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 1 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)


def sort_alphanum(a):

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(a, key = alphanum_key)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.video_test_path, regx='.*.jpg', printable=False))


    #valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    all_files = [f for f in os.listdir('/home/best_student/Documents/SR_Joelle/project/frames_to_test') if f.endswith('.jpg')]
    sorted_files = sort_alphanum(all_files)
    for file in sorted_files:
        print(file)

    #valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.video_test_path, n_threads=32)
    valid_hr_imgs = tl.vis.read_images(sorted_files, path=config.VALID.video_test_path, n_threads=32)


    t_image = tf.placeholder('float32', [1, 720, 1280, 6], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###

    # Warmup on a dummy image
    im_warmup = 0.2 * np.ones((720, 1280, 6), dtype=np.uint8)
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [im_warmup]})
    print("warm up took: %4.4fs" % (time.time() - start_time))

    ###========================== DEFINE MODEL ============================###

    print(len(valid_hr_imgs))
    for i in range(0,len(valid_hr_imgs)):
        indices_1 = [i,i+1]
        train_vid_img_list_s1 = [valid_hr_imgs[j] for j in indices_1]

        train_vid_seqs =[np.concatenate([train_vid_img_list_s1[0], train_vid_img_list_s1[1]], 2)]

        train_vid_seqs = np.asarray(train_vid_seqs)
        #mod_0 = i*3
        mod_1 = i*2 + 1
        #mod_2 = i*3 + 2

        #(1, 1080, 1920, 6)

        train_vid_seqs = (train_vid_seqs / 127.5) - 1

        size = train_vid_seqs.shape

        ### hereeeeeee
        start_time = time.time()
        #while 1 == 1:
        #out = sess.run(net_g.outputs, {t_image: [valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img]})

        out = sess.run(net_g.outputs, {t_image: train_vid_seqs})
        print("Frame %d took: %4.4fs" %(mod_1, (time.time() - start_time)))

        print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/frame_%d.jpg' %mod_1)


def validate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    train_video_folders = '/media/saeed-lts5/Data-Saeed/SuperResolution/youtube8m-dataset/frames'
    train_vid_list = sorted(tl.files.load_folder_list(path=train_video_folders))

    train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[30] + '/frames/', regx='.*.png', printable=False))

    #print('len train_vid_img_list')
    #print(len(train_vid_img_list)) # 150
    #print('len train_vid_list')
    #print(len(train_vid_list)) # 6757

    train_vid_list = train_vid_list[5020:5040]
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    """
    all_files = [f for f in os.listdir('/home/best_student/Documents/SR_Joelle/project/frames_to_test') if f.endswith('.jpg')]
    sorted_files = sort_alphanum(all_files)
    for file in sorted_files:
        print(file)
    """
    #valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.video_test_path, n_threads=32)
    #valid_hr_imgs = tl.vis.read_images(sorted_files, path=config.VALID.video_test_path, n_threads=32)

    t_image = tf.placeholder('float32', [1, 240, 300, 6], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)
    #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_safe.npz', network=net_g)

    ###======================= EVALUATION =============================###

    # Warmup on a dummy image
    im_warmup = 0.2 * np.ones((240, 300, 6), dtype=np.uint8)
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [im_warmup]})
    print("warm up took: %4.4fs" % (time.time() - start_time))


    ###========================== DEFINE MODEL ============================###

    for i in range(0,len(train_vid_list)):

        train_vid_img_list = sorted(tl.files.load_file_list(path=train_vid_list[i] + '/frames/', regx='.*.png', printable=False))
        #b_imgs_384 = tl.vis.read_images([train_vid_img_list[110]], path=train_vid_list[100] + '/frames/', n_threads=32)
        indices_1 = [19,21]
        train_vid_img_list_s1 = [train_vid_img_list[j] for j in indices_1]

        #train_target_vid_imgs = tl.vis.read_images([train_vid_img_list[20],train_vid_img_list[50],train_vid_img_list[80],train_vid_img_list[110]], path=train_vid_list[0] + '/frames/', n_threads=32)

        train_vid_img_list_s1 = tl.vis.read_images(train_vid_img_list_s1,path=train_vid_list[i] + '/frames/', n_threads=32)

        target = tl.vis.read_images([train_vid_img_list[20]],path=train_vid_list[i] + '/frames/', n_threads=32)
        first_frame = tl.vis.read_images([train_vid_img_list[19]],path=train_vid_list[i] + '/frames/', n_threads=32)
        second_frame = tl.vis.read_images([train_vid_img_list[21]],path=train_vid_list[i] + '/frames/', n_threads=32)

        #print('shape is')
        #print(np.shape(target))

        target = tl.prepro.threading_data(target, fn = crop_sub_imgs_fn_2, is_random = False)
        first_frame = tl.prepro.threading_data(first_frame, fn = crop_sub_imgs_fn_2, is_random = False)
        second_frame = tl.prepro.threading_data(second_frame, fn = crop_sub_imgs_fn_2, is_random = False)
        print('cropped')
        #target = (255. / 2.) * target
        #target = target.astype(np.uint8)

        tl.vis.save_image(target[0,:,:,:], save_dir + '/target_%d.png' %i)
        tl.vis.save_image(first_frame[0,:,:,:], save_dir + '/first_frame_%d.png' %i)
        tl.vis.save_image(second_frame[0,:,:,:], save_dir + '/second_frame_%d.png' %i)
        print('saved')


        train_vid_img_list_s1 = tl.prepro.threading_data(train_vid_img_list_s1, fn = crop_sub_imgs_fn_2, is_random=False)

        train_vid_seqs =[np.concatenate([train_vid_img_list_s1[0], train_vid_img_list_s1[1]], 2)]

        train_vid_seqs = np.asarray(train_vid_seqs)
        #mod_0 = i*3
        #mod_1 = i*2 + 1
        #mod_2 = i*3 + 2

        #(1, 360, 640, 6)

        #train_vid_seqs = (train_vid_seqs / 127.5) - 1

        size = train_vid_seqs.shape
        print('size vid seq is')
        print(size)

        size = np.shape(target)
        print('size target is')
        print(size)


        ### hereeeeeee
        start_time = time.time()
        #while 1 == 1:
        #out = sess.run(net_g.outputs, {t_image: [valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img, valid_lr_img]})

        out = sess.run(net_g.outputs, {t_image: train_vid_seqs})
        print("Frame %d took: %4.4fs" %(i, (time.time() - start_time)))

        print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/frame_%d.png' %i)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    elif tl.global_flag['mode'] == 'validate':
        validate()
    else:
        raise Exception("Unknow --mode")
