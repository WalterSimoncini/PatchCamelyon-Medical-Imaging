def show(imgs: List[torch.tensor] | torch.tensor, labels: List[str | int] = None):
    if type(imgs) != list:
        imgs = [imgs]

    if labels is None:
        labels = [""] * len(imgs)

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(12, 6))

    for i, (img, label) in enumerate(zip(imgs, labels)):
        img = img.detach()
        img = F.to_pil_image(img)

        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_xlabel(label)
