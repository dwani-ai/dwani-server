sudo apt-get update -y
sudo apt-get upgrade -y

sudo apt-get install -y python3-venv
sudo apt-get install -y python3-pip
sudo apt-get install -y ffmpeg
sudo apt install net-tools -y


python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

#cd src/asr_indic_server
python src/asr_api.py

