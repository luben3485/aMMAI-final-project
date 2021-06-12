import gdown

# relation 5-way 5-shot 0612
id = '1u_f_7uO8sO6yDlUMWlpujPlF37Gra7HF'
url = 'https://drive.google.com/uc?id=' + id
output = 'checkpoints.zip'
gdown.download(url, output)
