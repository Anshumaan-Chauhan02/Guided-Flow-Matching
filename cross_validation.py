#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def cross_val(lr_list, min_delta, patience, epochs, flow_train_dataset, flow_val_dataset, batch_size, reshape_text, conditional_gen=False, epoch_print, infer_num, epoch_save, final_infer_num):
  for i in range(len(lr_list)):
    print(f'Learning Rate: {lr_list[i]}')
    loss_vals, model = train_one_epoch(min_delta, patience, epochs, flow_train_dataset, lr_list[i], batch_size, reshape_text, convnet= True, conditional_gen, epoch_print, infer_num, epoch_save, final_infer_num)
    fid = FrechetInceptionDistance(normalize=True)
    imgs = []
    captions = []
    real_imgs = []
    if not os.path.exists("/content/drive/MyDrive/CS 682/CS682 Project/eval_models/conditional_unet/"):
        os.makedirs("/content/drive/MyDrive/CS 682/CS682 Project/eval_models/conditional_unet/")
    if not os.path.exists("/content/drive/MyDrive/CS 682/CS682 Project/eval_models/unconditional_unet/"):
        os.makedirs("/content/drive/MyDrive/CS 682/CS682 Project/eval_models/unconditional_unet/")
    print("Started Evaluation")
    for j in range(len(flow_val_dataset)):
      if conditional_gen:
        x_1, label, caption, x_0 = flow_val_dataset[j]
        captions.append(caption)
        real_imgs.append(torch.FloatTensor(x_1))
        img = inference(model, caption, reshape_text, True)
        img = (img - torch.min(img))/(torch.max(img) - torch.min(img))
      else:
        x_0, x_1 = flow_val_dataset[j]
        real_imgs.append(x_1)
        img = inference(model)
      imgs.append(img.cpu())
    print("Converting to Torch")
    real_imgs = torch.from_numpy(np.stack(real_imgs, axis=0).reshape(len(real_imgs),3,32,32))
    imgs = torch.from_numpy(np.stack(imgs, axis=0).reshape(len(imgs),3,32,32))
    fid.update(real_imgs, real=True)
    fid.update(imgs, real=False)
    print("Started computing FID")
    print(f"FID: {float(fid.compute())}")
    if conditional_gen:
      clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
      clip_score_val = clip_score_fn(imgs, captions).detach()
      clip_score_val = round(float(clip_score_val), 4)
      print(f'CLIP Score Value: {clip_score_val}')
      torch.save(model, f'/content/drive/MyDrive/CS 682/CS682 Project/eval_models/conditional_unet/unet_{i}.pth')
    else:
      torch.save(model, f'/content/drive/MyDrive/CS 682/CS682 Project/eval_models/unconditional_unet/unet_{i}.pth')

    plot_loss(loss_vals)

    print('----------------------------------------------------')

