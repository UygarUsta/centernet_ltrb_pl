Reimplementation of Custom Centernet Model.

Uses CIOU loss, model is Mobilenetv4 with FPN. Mosaic Augmentation is implemented, along with MixUp. No offset head, data is encoded as ltrb. 

Data directory should be like:

DataRoot:
         train_images:
                     1.jpg
                     1.xml 
         val_images:
                    1.jpg
                    1.xml


Make sure to delete classes.txt and .json files if you want to train with another dataset.
