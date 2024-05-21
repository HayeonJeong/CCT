import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

# 데이터셋을 다운로드할 디렉토리를 지정합니다.
download_root = './data'

# CIFAR-10 데이터셋을 로드하고 이미지를 변환합니다.
transform = transforms.Compose([
    transforms.ToTensor()
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 테스트 데이터셋을 로드합니다.
test_dataset = datasets.CIFAR10(root=download_root, train=False, download=True, transform=transform)

# 저장할 디렉토리를 생성합니다.
save_dir = './test_images'
os.makedirs(save_dir, exist_ok=True)

# 처음 100개의 이미지를 가져와서 PNG 파일로 저장합니다.
for i in range(100):
    image, label = test_dataset[i]
    filename = os.path.join(save_dir, f'image_{i+2}.png')
    transforms.ToPILImage()(image).save(filename)

print('처음 100개의 이미지가 PNG 파일로 저장됨')
