# Importing data

1. Create a Kaggle account
2. In your account settings, create an "API Token". This should add a kaggle.json file.
3. In your terminal run: pip install kaggle
4. From there follow these instructions:
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/ <- Your local downloads
   chmod 600 ~/.kaggle/kaggle.json
5. Once you have added the kaggle.json setup, go to the datasets directory and run the following:
6. kaggle competitions download -c march-machine-learning-mania-2025
7. unzip kaggle competitions download -c march-machine-learning-mania-2025
