# This is a utility script to help download the Diabetic Retinopathy data
# from Kaggle. You should have the kaggle-api installed and configured
# on your computer. Download here: https://github.com/Kaggle/kaggle-api
# The dataset is 8Gb of retinal images, each of them labelled on a scale
# of 0-4 of how bad they have diabetic retinopathy.
#
# You will also have to accept the competition rules here:
# https://www.kaggle.com/c/aptos2019-blindness-detection

mkdir -p data/

echo -e "\nDownloading dataset."
kaggle competitions download -c aptos2019-blindness-detection -f train.csv -p data
kaggle competitions download -c aptos2019-blindness-detection -f train_images.zip

echo -e "\nExtracting files."
unzip -q train_images.zip -d data/images

echo -e "\nCleaning up files."
rm train_images.zip
