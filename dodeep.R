
#https://www.tensorflow.org/install/gpu

#https://tensorflow.rstudio.com/tutorials/advanced/
#https://ecostat.gitlab.io/imaginecology/basics.html
library(keras)
library(tensorflow)
library(reticulate)
#library(tfdatasets)
library(dplyr)
library(tidyr)
library(ggplot2)
library(flextable)

#define the right python
#Sys.setenv(RETICULATE_PYTHON="/usr/bin/python3.6")
#install_tensorflow()
#install_keras()
#install_tensorflow(extra_packages="SciPy")



refall<-read.csv("../datcells_readings_fin.csv")
#create the right directory organization
#read images info
ref<-read.csv("../datcells_readings_fin.csv")%>%
	transmute(cell_pic,cell_type,slideid)%>%
	filter(!is.na(cell_type))%>%
	mutate(path=paste0("../LEPI_WHI_cells/",gsub("LEPI_","",slideid),"/",cell_pic))%>%
	mutate(destpath=paste0("./data/",cell_type,"/"))
ref0<-read.csv("../datcells_readings_fin.csv")%>%
	transmute(cell_pic,cell_type,slideid)%>%
	mutate(ind=fct1(cell_pic))
fct1<-function(a){strsplit(a,split="-")[[1]][1]}
ref0$ind<- unlist(lapply(ref0$cell_pic,fct1))
	#filter(!is.na(cell_type))%>%
ref0<-ref0%>%mutate(path=paste0("../LEPI_WHI_cells/",ind,"/",cell_pic))%>%
	mutate(destpath=paste0("./test/test"))

#ref table 1 
#ref table 1 
ref2<-read.csv("../datcells_readings_fin.csv")%>%
	transmute(ind=substr(cell_pic,1,7),cell_type=replace_na(cell_type,"nodata"))
nomtmp<- c("Atretic oocyte alpha",
		  "Cortical alveoli oocyte",
		  "Oogonium",
		  "Primary oocyte stage 1",
		  "Primary oocyte stage 2",
		  "Vittelogenic oocyte phase 1",
		  "Undetermined",
		  "Not read")
nomcell<-data.frame(cell_type=c("aoA","cao","og","po1","po2","vtg1","U","nodata"),
	   cell=factor(nomtmp,levels=nomtmp))
ref2<-left_join(ref2,nomcell)%>%
	group_by(cell_type=cell,ind)%>%summarise(n=n())%>%
	pivot_wider(values_from=n,names_from=ind,values_fill=0)%>%
	ungroup()
fref2<-ref2%>%flextable()%>%
		align(j=names(ref2),align="center")%>%#,part="all")%>%
		color(i=8,color="blue")%>%
		bold(i=7:8)%>%
		theme_box()%>%
		autofit()



#directories
dir.create("data")
dir.create("test")
dir.create("test/test")
dirtype<-paste0("./data/",unique(ref$cell_type))
lapply(dirtype,dir.create)
#copy img to the righ dir
mapply(file.copy,ref$path,ref$destpath)
#create test dir with aaaaalll images
file.copy(ref0$path,"test/test")


#create the img dataset in keras world
#from https://www.r-bloggers.com/2021/03/how-to-build-your-own-image-recognition-app-with-r-part-1/
label_list <- dir("./data/")
output_n <- length(label_list)
save(label_list, file="label_list.R")

width <- 173 
height<- 173
target_size <- c(width, height)
rgb <- 3 #color channels
path_train <- "./data/"
train_data_gen <- image_data_generator(rescale = 1/255, validation_split = .2)
train_images <- flow_images_from_directory(path_train,
  train_data_gen,
  subset = 'training',
  target_size = target_size,
  class_mode = "categorical",
  shuffle=F,
  classes = label_list,
  seed = 2021)
validation_images <- flow_images_from_directory(path_train,
 train_data_gen, 
  subset = 'validation',
  target_size = target_size,
  class_mode = "categorical",
  classes = label_list,
  seed = 2021)
table(train_images$classes)
plot(as.raster(train_images[[1]][[1]][17,,,]))

#model 
#load imagenet model
mod_base <- application_xception(weights = 'imagenet', 
   include_top = FALSE, input_shape = c(width, height, 3))
freeze_weights(mod_base) 

model_function <- function(learning_rate = 0.001, 
  dropoutrate=0.2, n_dense=1024){
  k_clear_session()
  model <- keras_model_sequential() %>%
    mod_base %>% 
    layer_global_average_pooling_2d() %>% 
    layer_dense(units = n_dense) %>%
    layer_activation("relu") %>%
    layer_dropout(dropoutrate) %>%
    layer_dense(units=output_n, activation="softmax")
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  return(model)
}

model<-model_function()
model

#training model
batch_size <- 32
epochs <- 20 

hist <- model %>% fit(
  train_images,
  steps_per_epoch = train_images$n %/% batch_size, 
  epochs = epochs, 
  validation_data = validation_images,
  validation_steps = validation_images$n %/% batch_size,
  verbose = 2
)

#save model
#save(model,file="model.rdata")
#load("model.rdata")
library(ggplot2)
plot(hist)
ggsave(file="loss_accuracy.png")
#evaluate and test
path_test <- "data"
test_data_gen <- image_data_generator(rescale = 1/254)
test_images <- flow_images_from_directory(path_test,
   test_data_gen,
   target_size = target_size,
   class_mode = "categorical",
   classes = label_list,
   shuffle = F,
   seed = 2021)

model %>% evaluate_generator(test_images,steps=test_images$n)


save_model_hdf5(model,"my_model.h5")
model<-load_model_hdf5("my_model.h5")

#single image test
test_image <- image_load("./test/WHI_388_V-11.jpeg",target_size = target_size)
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred<-model%>%predict(x)
pred2<-model%>%predict(test_images)

path_test <- "./test/"
test_data_gen <- image_data_generator(rescale = 1/254)
test_images <- flow_images_from_directory(path_test, test_data_gen,
   #target_size = target_size,
   class_mode = "categorical",
   #classes = label_list,
   shuffle = F,
   seed = 2021)

model %>% evaluate_generator(test_images,steps=test_images$n)

pred<-model%>%predict(test_images)
save(pred,file="pred.rdata")

rez<-cbind(data.frame(name=test_images$filenames),pred)%>%
	mutate(name=gsub('test/','',name))
ref2<-read.csv("../datcells_readings_fin.csv")
rez<-full_join(rez,ref2,by=c("name"="cell_pic"))
names(rez)[2:8]<-label_list
rez[2:8]<-round(rez[2:8]*100,2)

#add pred classe to original files
library(magick)
image_ggplot(magick::image_read(paste0("./test/test/",rez$name[1])))

plot(as.raster(test_images[[1]][[1]][1,,,]))

r0<-(magick::image_read(paste0("./test/test/",rez$name[1])))
qplot(mpg, wt, data = mtcars)+annotation_raster(r0)#,25,30,3,5)

rez2<-rez%>%select(file=name,cell_type,m.majoraxis,aoA:vtg1)%>%pivot_longer(aoA:vtg1)%>%
	group_by(file,cell_type)%>%slice(which.max(value))%>%
	ungroup()%>%mutate(x=jitter(as.numeric(factor(cell_type)),factor=2),
			   y=jitter(as.numeric(factor(name)),factor=2),
			   path=paste0('./test/test/',file))%>%
	filter(!is.na(cell_type))#%>%
	#group_by(cell_type)%>%
	#sample_n(2,replace=T)
	#filter(m.majoraxis>300)
rez2ok<-rez2%>%filter(name==cell_type,m.majoraxis>50)%>%
	group_by(cell_type)%>%
	sample_n(2,replace=F)
rez2ko<-rez2%>%filter(name!=cell_type,m.majoraxis>50)%>%
	group_by(name)%>%
	sample_n(1,replace=T)
rez2all<-rbind(rez2ko,rez2ok)
#convert to png with transparency
for(i in 1:nrow(rez2all)){
	rez2all$path2[i]<-gsub("./test/test","pipo",gsub("jpeg","png",rez2all$path[i]))
	cmd<-paste0("convert ",rez2all$path[i]," -transparent white ",rez2all$path2[i])
	system(cmd)
}
library(ggimage)
ggplot(rez2all%>%filter(!is.na(cell_type)),aes(x=cell_type,y=name))+
	geom_image(aes(image=path2),position=position_jitter(width=.3,height=.3),size=0.2)+
	xlab("Cell type observed")+ylab("Cell type predicted")+
	theme_bw()

rez3<-rez%>%select(file=name,cell_type,m.majoraxis,aoA:vtg1)%>%pivot_longer(aoA:vtg1)%>%
	group_by(file,cell_type)%>%slice(which.max(value))%>%
	ungroup()%>%group_by(cell_type,name)%>%summarise(n=n())%>%
	mutate(cell_type=replace_na(cell_type,"Not observed"))%>%
	ungroup()

ggplot(rez3,aes(x=name,y=n,fill=name))+
	geom_bar(stat="identity",position="dodge")+
	facet_wrap(~cell_type,scale="free")+
	xlab("Cell type predicted")+ylab("Number of cells")+
	labs(fill="Cell type Predicted")+
	theme_bw()

image(aes(image=path2),position=position_jitter(width=.3,height=.3),size=0.2)+
	xlab("Cell type observed")+ylab("Cell type predicted")+
	theme_bw()
	
	
	
	mutate(x=jitter(as.numeric(factor(cell_type)),factor=2),
			   y=jitter(as.numeric(factor(name)),factor=2),
			   path=paste0('./test/test/',file))

#single image test
test_image <- image_load("./test/WHI_388_V-11.jpeg",target_size = target_size)
x <- image_to_array(test_image)
x <- array_reshape(x, c(1, dim(x)))
x <- x/255
pred<-model%>%predict(x)
pred2<-model%>%predict(test_images)

pred3<-predict(model)




stop()


train_images<-flow_images_from_directory("./data",
	train_data_gen,subset="training",target_size=targe)







pipo<-ref[grepl("WHI_472_V-8.jpeg",ref$cell_pic),]




#read the image
#ref file

#test python path check
list_ds<-file_list_dataset(as.vector(ref$path))
list_ds%>%reticulate::as_iterator()%>%reticulate::iter_next()

#plot(magick::image_read(pipo$path))
#convert file path to (image_data_label) pair
get_label<-function(path,ref){
	tmp<-ref$cell_type[ref$path==path]
	tmp%>%tf$equal(tmp)%>%tf$cast(dtype=tf$float32)
}
get_label(list_ds$path,ref)

decode_img <- function(file_path, height = 173, width = 173) {
file_path<-pipo$path;height=173;width=173
  size <- as.integer(c(height, width))
  file_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$convert_image_dtype(dtype = tf$float32) %>% 
    tf$image$resize(size = size)
}

decode_img(pipo$ref)

preprocess_path <- function(file_path) {
  list(
    decode_img(file_path),
    get_label(file_path)
  )
}

# num_parallel_calls are going to be autotuned
labeled_ds <- ref$path%>% 
  dataset_map(preprocess_path, num_parallel_calls = tf$data$experimental$AUTOTUNE)

preprocess_path(pipo$ref)






pred3<-predict(model)




stop()


train_images<-flow_images_from_directory("./data",
	train_data_gen,subset="training",target_size=targe)







pipo<-ref[grepl("WHI_472_V-8.jpeg",ref$cell_pic),]




#read the image
#ref file

#test python path check
list_ds<-file_list_dataset(as.vector(ref$path))
list_ds%>%reticulate::as_iterator()%>%reticulate::iter_next()

#plot(magick::image_read(pipo$path))
#convert file path to (image_data_label) pair
get_label<-function(path,ref){
	tmp<-ref$cell_type[ref$path==path]
	tmp%>%tf$equal(tmp)%>%tf$cast(dtype=tf$float32)
}
get_label(list_ds$path,ref)

decode_img <- function(file_path, height = 173, width = 173) {
file_path<-pipo$path;height=173;width=173
  size <- as.integer(c(height, width))
  file_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$convert_image_dtype(dtype = tf$float32) %>% 
    tf$image$resize(size = size)
}

decode_img(pipo$ref)

preprocess_path <- function(file_path) {
  list(
    decode_img(file_path),
    get_label(file_path)
  )
}

# num_parallel_calls are going to be autotuned
labeled_ds <- ref$path%>% 
  dataset_map(preprocess_path, num_parallel_calls = tf$data$experimental$AUTOTUNE)

preprocess_path(pipo$ref)







fashion_mnist <- dataset_fashion_mnist()

c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test




library(tidyverse)
library(platypus)
library(abind)

test_yolo <- yolo3(
  net_h = 416, # Input image height. Must be divisible by 32
  net_w = 416, # Input image width. Must be divisible by 32
  grayscale = FALSE, # Should images be loaded as grayscale or RGB
  n_class = 80, # Number of object classes (80 for COCO dataset)
  anchors = coco_anchors # Anchor boxes
)


