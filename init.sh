# process imagenet
# train
echo "Extract Train set of ImageNet"
cd data/image-net &&
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train &&
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..
# val
echo "Extract Val set of ImageNet"
mkdir val && mv ILSVRC2012_img_val.tar val/ && mv valprep.sh val && cd val &&
tar -xvf ILSVRC2012_img_val.tar &&
cat valprep.sh | bash
cd ..

# change dir back to main directory
cd ../..

