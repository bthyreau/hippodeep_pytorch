import matplotlib.pyplot as plt
import nibabel
import scipy.ndimage
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
import scipy.ndimage
import matplotlib

matplotlib.use("Agg")


pal = LinearSegmentedColormap.from_list("test", ["darkblue", "aqua"], N=25)
pal = LinearSegmentedColormap.from_list("test", ["orangered", "gold"], N=25)
#palB = "prism" 
palB = LinearSegmentedColormap.from_list("test", ["darkblue", "aqua"], N=8)
palalpha = LinearSegmentedColormap.from_list("test", [(0,0,0,0), "orangered", "orange",], N=25)

def circle_mask_transparency(slice_data, center, radius):
    m = ((np.indices(slice_data.shape) - np.reshape(center, (2,1,1)))**2).sum(0) * 1.
    m[m <= radius ** 2] = 1.
    m[m > radius ** 2] = np.nan
    return m

def circle_mask_smooth_transparency(slice_data, center, radius):
    m = ((np.indices(slice_data.shape) - np.reshape(center, (2,1,1)))**2).sum(0)
    rmin, rmax = (radius - 20)**2, radius**2
    m_smooth = 1 - (m.clip(rmin, rmax) - rmin) / (rmax - rmin)
    m_smooth[m > rmax] = np.nan
    return m_smooth

def generate_images(fname, rendering_mode=["circle-png", "circle-jpg", "plain"]):

    fnames = {}

    assert ".nii.gz" in fname
    subjid = fname.replace(".nii.gz","") # e.g. test/output_t1_papaya.nii.gz

    outpth = fname.replace(".nii.gz","_thumbnails")

    img = nibabel.load("%s.nii.gz" % subjid)
    if os.path.exists("%s_mask_LR.nii.gz" % subjid):
        imgH = nibabel.load("%s_mask_LR.nii.gz" % subjid)
    else:
        imgH = nibabel.Nifti1Image(np.zeros(img.shape, "uint8"), img.affine)

    assert img.shape == (165, 253, 238)

    hippo_coloralpha = .75
    cortex_colorpalpha = .25

    # papaya box is 165 253 238, with R hippo approximately at 60 125 122


## Main report slices images
##
    if 1:
        # Coronal, Circle-Centered
        slice_data = np.asarray(img.dataobj)[:,125,:]
        m = circle_mask_smooth_transparency(slice_data, center=[82, 122], radius=100)
        slice_data = (slice_data * m)[:, 125-100: 125+100]
        slice_label = np.asarray(imgH.dataobj).astype(np.float32)[:,125,:]
        slice_label = slice_label[:, 125-100: 125+100]

        import scipy.ndimage
        slice_label_smooth = scipy.ndimage.gaussian_filter(slice_label, (7,7))
        slice_label_smooth[slice_label_smooth < 16] = np.nan
        slice_label_smooth[slice_label_smooth > 32] = np.nan

        slice_label[slice_label < 64] = np.nan

        fig, ax = plt.subplots(figsize=(165 / 30., 200 / 30.))  # Create a new figure and axes instance with the specified figure size
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.imshow(np.rot90(slice_data), cmap="gray", aspect="auto", interpolation="bilinear", origin="upper", vmin=np.nan_to_num(slice_data.min()), vmax=np.percentile(np.nan_to_num(slice_data), 90) * 1.2)
        ax.imshow(np.rot90(slice_label), cmap=pal, aspect="auto", interpolation="bilinear", origin="upper", alpha=hippo_coloralpha)
        if "circle-png" in rendering_mode:
            fig.savefig(f"{outpth}_cor_circle.png", transparent=True, dpi=48)
        if "circle-jpg" in rendering_mode:
            fig.savefig(f"{outpth}_cor_circle.jpg", dpi=48)
        if "plain" in rendering_mode:
            fig.savefig(f"{outpth}_cor_black.jpg", facecolor='black', dpi=48)

        ax.images[-1].remove()
        plt.close(fig)

    if 1:
        # Axial, Circle-Centered
        slice_data = np.asarray(img.dataobj)[:,:,122]
        m = circle_mask_smooth_transparency(slice_data, center=[82, 122], radius=100)

        slice_data = (slice_data * m)[:, 122-100: 122+100]
        slice_label = np.asarray(imgH.dataobj).astype(np.float32)[:,:,122]
        slice_label = slice_label[:, 122-100: 122+100]

        import scipy.ndimage
        slice_label_smooth = scipy.ndimage.gaussian_filter(slice_label, (7,7))
        slice_label_smooth[slice_label_smooth < 16] = np.nan
        slice_label_smooth[slice_label_smooth > 32] = np.nan

        slice_label[slice_label < 64] = np.nan

        fig, ax = plt.subplots(figsize=(165 / 30., 200 / 30.))  # Create a new figure and axes instance with the specified figure size
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.imshow(np.rot90(slice_data), cmap="gray", aspect="auto", interpolation="bilinear", origin="upper", vmin=np.nan_to_num(slice_data.min()), vmax=np.percentile(np.nan_to_num(slice_data), 90) * 1.2)
        ax.imshow(np.rot90(slice_label), cmap=pal, aspect="auto", interpolation="bilinear", origin="upper", alpha=hippo_coloralpha)
        if "circle-png" in rendering_mode:
           fig.savefig(f"{outpth}_ax_circle.png", transparent=True, dpi=48)
        if "circle-jpg" in rendering_mode:
           fig.savefig(f"{outpth}_ax_circle.jpg", dpi=48)
        if "plain" in rendering_mode:
           fig.savefig(f"{outpth}_ax_black.jpg", facecolor='black', dpi=48)
 
        ax.images[-1].remove()
        plt.close(fig)

    if 1:
        # Sagittal, Circle-Centered
        slice_data = np.asarray(img.dataobj)[60,:,:]

        m = circle_mask_smooth_transparency(slice_data, center=[125, 122], radius=100)

        slice_data = (slice_data * m)[125-100:125+100, 122-100: 122+100]
        slice_label = np.asarray(imgH.dataobj).astype(np.float32)[60,:,:]
        slice_label = slice_label[125-100:125+100, 122-100: 122+100]

        import scipy.ndimage
        slice_label_smooth = scipy.ndimage.gaussian_filter(slice_label, (7,7))
        slice_label_smooth[slice_label_smooth < 16] = np.nan
        slice_label_smooth[slice_label_smooth > 32] = np.nan

        slice_label[slice_label < 64] = np.nan


        fig, ax = plt.subplots(figsize=(200 / 30., 200 / 30.))  # Create a new figure and axes instance with the specified figure size
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1, 0, 0)
        ax.imshow(np.rot90(slice_data), cmap="gray", aspect="auto", interpolation="bilinear", origin="upper", vmin=np.nan_to_num(slice_data.min()), vmax=np.percentile(np.nan_to_num(slice_data), 90) * 1.2)
        ax.imshow(np.rot90(slice_label), cmap=pal, aspect="auto", interpolation="bilinear", origin="upper", alpha=hippo_coloralpha)
        if "circle-png" in rendering_mode:
            fig.savefig(f"{outpth}_sag_circle.png", transparent=True, dpi=48)
        if "circle-jpg" in rendering_mode:
            fig.savefig(f"{outpth}_sag_circle.jpg", dpi=48)
        if "plain" in rendering_mode:
            fig.savefig(f"{outpth}_sag_black.jpg", facecolor='black', dpi=48)
        ax.images[-1].remove()
        plt.close(fig)

if __name__ == "__main__":
    import os, sys
    fname = sys.argv[1]
    generate_images(fname, os.path.dirname(fname), rendering_mode)
