import face_recognition as face_rec
from pathlib import Path
from loguru import logger
from transliterate import translit
from threading import Thread
from time import sleep
from datetime import datetime
from timedelta import Timedelta as TD
import cv2
import pickle
import numpy
from PIL import Image


logger.add(Path('Data', 'LogFile.log'), level='DEBUG')
users_list = pickle.loads(open(Path('Data', 'face_rec.db'), "rb").read())# {"encodings": [encoding], "names": [user_name], 'user_info': [user_info]}
cam = cv2.VideoCapture(0)
stop = False


def update_users():
    global users_list
    users_list = pickle.loads(open(Path('Data', 'face_rec.db'), "rb").read()) 

def main_thread():
    cache_names = []
    time = datetime.now()
    time2 = datetime.now()
    while stop:
        image = cam.read()[1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if datetime.now() > time2:
            face_pos = face_rec.face_locations(rgb, model='hog')
        else:
            face_pos = []
        if face_pos:
            names = []
            if not cache_names or datetime.now() > time:
                encoding = face_rec.face_encodings(rgb)
                for encoding in encoding:
                    name = 'Unreg. user'
                    for i, elem in enumerate(users_list['encodings']):
                        if face_rec.compare_faces(encoding, elem, tolerance=0.5)[0]:
                            name = translit(users_list['names'][i], 'uk', reversed=True).replace('Э', 'E').replace('э', 'e')
                    names.append(name)
                cache_names = names
                if 'Unreg. user' in cache_names:
                    time = datetime.now() + TD(seconds=2)
                else:
                    time = datetime.now() + TD(seconds=6)
            else:
                names = cache_names
            for index, (top, right, bottom, left) in enumerate(face_pos):
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(image, names[index], (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        else:
            cache_names = []
            time2 = datetime.now() + TD(seconds=2)
        cv2.imshow("Main", image)
        ch = cv2.waitKey(5)
        if ch == 27:
            break



def make_face_encoding() -> list:
    image = cam.read()[1]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_encoding = face_rec.face_encodings(rgb)
    if len(face_encoding) > 0:
        return face_encoding[0]
    return []



def save_user(user_name, user_info='10-Б класс', attempt = 0):
    if not user_name in users_list['names']:
        image = cam.read()[1]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoding = face_rec.face_encodings(rgb)
        if len(encoding) > 0:
            with open(Path('Data', 'face_rec.db'), 'wb') as file:
                users_list['encodings'].append(encoding)
                users_list['names'].append(user_name)
                users_list['user_info'].append(user_info)
                file.write(pickle.dumps({"encodings": users_list['encodings'], "names": users_list['names'], 'user_info': users_list['user_info']}))
            logger.success(f'New user successfully saved: {user_name}')
            return
        if attempt == 0:
            logger.critical('Человек не распознан, поменяйте позицию.')
            sleep(3)
            save_user(user_name, user_info='10-Б класс', attempt = 1)
        else:
            logger.critical('Человек снова не распознан')
    logger.info('Данный человек уже имеется в базе')


def main():
    global stop
    logger.info('[*] Successfully inited')
    while True:
        task = input('Вы в главном меню\n1) Зарегистрировать человека\n2) Запустить/остановить работу программы\n3) Удалить человека\n\n')
        if task == '1':
            user_name = input('Введите Имя человека для добавления в базу(0 для отмены): ')
            if not user_name == '0':
                save_user(user_name)
                update_users()
            else:
                print('Неверное имя')
        elif task == '2':
            if stop is True:
                stop = False
                print('Поток остановлен')
            else:
                stop = True
                Thread(target=main_thread).start()
                print('Поток запущен')
        
        elif task == '3':
            form = ''.join(f'{index+1}) {elem}\n' for index, elem in enumerate(users_list['names']))
            print(form)
            num = int(input('Введите номер человека(для отмены 0): '))-1
            if not num == -1 and num < len(users_list['names'])+1:
                print(f'Пользователь {users_list["names"][num]} удален')
                users_list['encodings'].pop(num)
                users_list['names'].pop(num)
                users_list['user_info'].pop(num)
                open(Path('Data', 'face_rec.db'), 'wb').write(pickle.dumps({"encodings": users_list['encodings'], "names": users_list['names'], 'user_info': users_list['user_info']}))
                update_users()

if __name__ == '__main__':
    main()