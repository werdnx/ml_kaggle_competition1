batch_size_ = batch_size = 2
# size of images
target_size_ = 380
epochs_ = 6

# hyperparams
label_smooth_fac = 0.05

iteration = '19'
model_name = 'model9_freeze_EfficientNetB3B4_gen_' + str(target_size_) + '_' + str(batch_size_) + '_' + str(
    epochs_) + '_' + iteration
