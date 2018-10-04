# Классификатор нейтрального лица

Задача - написать классификатор нейтрального лица, с упором на его дальнейшее использование в мобильном приложении в real-time. Предлагается в качестве базового варианта опираться на стандартную разметку 68 ключевых точек, которая использовалась в MULTI-PIE.
В рамках этого задания мы будем считать, что данная задача разбивается на две:
Детектор открытого рта
Детектор улыбки

Ограничения:
- не более 200 ms на фотографии с найденными ключевыми точками
- не более 1000 ms на поиск ключевых точек на фотографии
- не более 150 MB 
- общераспространенные зависимости (opencv, dlib, numpy, scipy, tensorflow, …) использовать можно, но они должны присутствовать в инструкции по установке
- языки: C++, Python

Ограничения по времени следует считать ориентировочными, можете их понимать как ограничения для конкретного компьютера, на котором вы будете выполнять задание (также допускается их поднятие в случае наличия у вас компьютера, далекого от современных стандартов).

В процессе работы разрешается любые общедоступные датасеты для обучения и/или тестирования. В случае использования 68 ключевых точек разрешается использовать готовые решения для их поиска (их размер не будет учитываться в ограничении на занимаемое место). 

Результатом вашей работы является программа, которая по набору путей до фотографий выводит 2 списка (по одному для каждого классификатора) фотографий, проходящих соответствующий фильтр. Также требуется предоставить программный код (лучше в виде приватного репозитория на github), поясняющую документацию для запуска и выборки, использованные для обучения/валидации моделей.

Плюсом будет являться описание того, какие метрики вы использовали при обучении/валидации классификаторов и почему, как бы вы подбирали параметры классификаторов в зависимости от возможных требований (например, при известной цене ошибки первого или второго рода).

Примеры фотографий можно найти здесь 
https://drive.google.com/file/d/1JcJGUX8NOkZvCUyxinxtkn4qjGD-DgQF/view?usp=sharing
(пароль к архиву "Iephohn9" без кавычек).

## Solution without landmark

    git clone https://gitlab.com/Danil328/software-engineer-neutral-face-task.git
    cd Solution without landmarks
    python main.py "path_to_images" "path_to_submission"
