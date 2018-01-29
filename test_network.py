import torch,sys
from util import util
from models import pretrained_networks as pn
from models import dist_model as dm
from IPython import embed



## Initializing the model
model = dm.DistModel()

# Linearly calibrated models
# model.initialize(model='net-lin',net='squeeze',use_gpu=True)
# model.initialize(model='net-lin',net='alex',use_gpu=True)
# model.initialize(model='net-lin',net='vgg',use_gpu=True)

# Off-the-shelf uncalibrated networks
# model.initialize(model='net',net='squeeze',use_gpu=True)
# model.initialize(model='net',net='alex',use_gpu=True)
# model.initialize(model='net',net='vgg',use_gpu=True)

# Low-level metrics
# model.initialize(model='l2',colorspace='Lab')
# model.initialize(model='ssim',colorspace='RGB')
# print('Model [%s] initialized'%model.name())

## Example usage with dummy tensors
# dummy_im0 = torch.Tensor(1,3,64,64) # image should be RGB, normalized to [-1,1]
# dummy_im1 = torch.Tensor(1,3,64,64)
# dist = model.forward(dummy_im0,dummy_im1)

## Example usage with images
ex_ref = util.im2tensor(util.load_image('./imgs/ex_ref.png'))
ex_p0 = util.im2tensor(util.load_image('./imgs/ex_p0.png'))
ex_p1 = util.im2tensor(util.load_image('./imgs/ex_p1.png'))
# ex_d0 = model.forward(ex_ref,ex_p0)[0]
# ex_d1 = model.forward(ex_ref,ex_ref)[0]

# ---------- new model -------
model.initialize(model='net-lin',net='alex',use_gpu=True)
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )


# ---------- new model -------
model.initialize(model='net-lin',net='vgg',use_gpu=True)
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )

# ---------- new model -------
model.initialize(model='net-lin',net='squeeze',use_gpu=True)
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )

# ---------- new model -------
model.initialize(model='net',net='alex',use_gpu=True)
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )


# ---------- new model -------
model.initialize(model='net',net='vgg',use_gpu=True)
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )

# ---------- new model -------
model.initialize(model='net',net='squeeze',use_gpu=True)
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )

# ---------- new model -------
model.initialize(model='l2',colorspace='Lab')
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )

# ---------- new model -------
model.initialize(model='ssim',colorspace='RGB')
print('Model [%s] initialized'%model.name())
same_image = model.forward(ex_ref,ex_ref)[0]
print("comparing Same Image : ",same_image )

my_image_load = util.im2tensor(util.load_image('./imgs/my_image.PNG'))
my_image = model.forward(my_image_load,my_image_load)[0]
print("comparing Same  My Image : ",same_image )

different_load = util.im2tensor(util.load_image('./imgs/different.jpg'))
different = model.forward(different_load,my_image_load)[0]
print("Comparing Two Different Images :",different,'\n' )