import matplotlib.pyplot as plt

def show_tensor_image(image_tensor):
    # reverse_transforms = transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #     transforms.Lambda(lambda t: t * 255.),
    #     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    #     transforms.ToPILImage(),
    # ])

    # Take first image of batch
    # Draw all images inside a tensor (N,W,H,C)
    image_np = image_tensor.detach().cpu().numpy()

    N = image_np.shape[0]
    plt.figure(figsize=(15,15*N))
    plt.axis('off')
    for idx in range(N):
        plt.subplot(1, N, idx+1)
        plt.imshow(image_np[idx,:,:,:])

def show_tensor_first_image(image_tensor):
    # reverse_transforms = transforms.Compose([
    #     transforms.Lambda(lambda t: (t + 1) / 2),
    #     transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    #     transforms.Lambda(lambda t: t * 255.),
    #     transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    #     transforms.ToPILImage(),
    # ])

    # Take first image of batch
    # Draw all images inside a tensor (N,C,W,H)

    image_np = image_tensor.detach().cpu().numpy()
    plt.figure(figsize=(5,5))
    plt.axis('off')
    plt.imshow(image_np[0,:,:,:])