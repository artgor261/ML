Добавьте в обучающие и тестовые данные столбец **modified_rubrics**, в котором будет то же, что и в **rubrics_id**, если соответствующая комбинация рубрик содержит хотя бы 100 заведений из обучающей (!) выборки, и строка **other** в противном случае.

Затем напишите могучий классификатор, который по заведению предсказывает медиану средних чеков среди тех в обучающей выборке, у которых с ним одинаковые **modified_rubrics** и город (вы спросите, почему медиану, а не самый частый -- спишем это на вдохновение; самый частый тоже можно брать - но медиана работает лучше).
