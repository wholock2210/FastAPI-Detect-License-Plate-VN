
# 🚗 FastAPI License Plate Detection API

A lightweight backend service built with **FastAPI** that detects and recognizes **Vietnamese vehicle license plates** from images using **YOLOv8**.

This project is designed as a RESTful API, making it easy to integrate license plate recognition into parking systems.

## 📌 Overview

This project provides an API for **automatic license plate detection** using deep learning.
It leverages the power of **YOLOv8** for object detection and runs on **FastAPI** for high-performance, asynchronous request handling.

Simply send an image to the API, and it will return the detected license plate information.



## Dependencies

* The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort).

```bash
git clone https://github.com/abewley/sort
```

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
conda create --prefix ./env python==3.10 -y
```

* Or use venv
``` bash
python3.10 -m venv .venv
```

* Activate the environment
``` bash
source activate ./env
``` 
* If you use fish

``` bash
source ./.venv/bin/activate.fish
```

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```

* [Uvicron](https://uvicorn.dev/)
```bash
pip install uvicorn
```

* Run the project using commands.
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

* Access the link
```bash
http://localhost:8000/docs#
```

## 🙏 Acknowledgements

This project is **adapted and extended** from the following open-source repository:

🔗 **Automatic License Plate Recognition using YOLOv8**  
Author: Muhammad Zeerak Khan  
GitHub: https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8
