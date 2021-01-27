import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.models import model_from_json
from sudoku import Sudoku
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle


def find_biggest_counter(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points

def split_to_boxes(img):
    rows = np.vsplit(img, 9)
    boxes=[]

    for r in rows:
        cols= np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)

    return boxes

def boxes_to_numbers(boxes):
    model = load_model('augusto-model')

    numbers = []

    for box in boxes:
        img = box[5:box.shape[0]-5, 5:box.shape[1]-5]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)

        prediction = model.predict(img)
        class_prediction = model.predict_classes(img)
        max_probability = np.amax(prediction)

        if max_probability > 0.9 and class_prediction[0] != 0:
            numbers.append(int(class_prediction[0]))
        else:
            numbers.append(0)

    return numbers

def paint_image(numbers, image, color = (0, 255, 0)):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2

    delta_w = int(img.shape[1]/9)
    delta_h = int(img.shape[0]/9)

    for i,number in enumerate(numbers):
        if number != 0:
            org = (int(i%9)*delta_w + int(delta_w/2) - 6, int(i/9)*delta_h + int(delta_h/2) + 12)
            img = cv2.putText(img, str(number), org, font,  font_scale, color, thickness, cv2.LINE_AA)

    return img

def solve_sudoku(numbers):
    board = [numbers[i:i+9] for i in range(0, 81, 9)]

    puzzle = Sudoku(3, board=board.copy())
    solved = puzzle.solve()
    solved_board = solved.board
    solved_numbers = []

    for i,row in enumerate(solved_board):
        for j,number in enumerate(row):
            if (board[i][j] == 0 or board[i][j] == None) and number != None:
                solved_numbers.append(number)
            else:
                solved_numbers.append(0)

    return solved_numbers

img_path = '5.jpeg' # Caminho da image para testar
height = 450 # Tamanho da altura resultante da mudança de perspectiva
width = 450 # Tamanho da largura resultante da mudança de perspectiva

### Processos básicos
img = cv2.imread(img_path)
img_debug = np.zeros((img.shape[0], img.shape[1], 3), np.uint8) # Usando só para preencher o imshow
img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
img_blur = cv2.GaussianBlur(img_gray.copy(), (7, 7), 0)  # Remoção de ruídos com gaussian blur
img_threshold = cv2.adaptiveThreshold(
    img_blur.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2) # Usando adaptive thresholding para achar os contornos

### Preparando imagens para o desenho dos contornos e do maior contorno encontrados
img_contours = img.copy()
img_bigcontour = img.copy()

### Encontrando e desenhando todos os contornos
contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

### Encontrando o maior contorno (o quadrado do sudoku)
biggest, maxArea = find_biggest_counter(contours)

if biggest.size != 0:
    ### Reordenando os pontos do contorno [(left, top), (right, top), (left, bottom), (right, bottom)]
    biggest = reorder(biggest)
    
    ### Desenhando o maior contorno
    cv2.drawContours(img_bigcontour, biggest, -1, (0, 0, 255), 25)

    ### Calculando a matrix de mudança de perspectiva usando os pontos do contorno localizado
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    ### Mudando a perpectiva reta com tamanho quadrado a partir da matrix calculada
    img_warp_colored = cv2.warpPerspective(img, matrix, (width, height))
    img_warp_gray = cv2.cvtColor(img_warp_colored,cv2.COLOR_BGR2GRAY)

    ### Pega a imagem na perspectiva reta e divide em 81 pequenas imagens (box)
    boxes = split_to_boxes(img_warp_gray)

    ### Fazendo o predict das boxes dos números usando um modelo de rede neural do keras treinado com mnist
    numbers = boxes_to_numbers(boxes)

    ### Usando os números preditos para resolver o sudoku com a biblioteca py-sudoku
    solved_numbers = solve_sudoku(numbers)

    ### Usado para mostrar o passo a passo no imshow
    numbers_painted = paint_image(numbers, img_warp_colored, (0, 0, 255))
    solved_painted = paint_image(solved_numbers, img_warp_colored, (0, 255, 0))
    img_warp_resized = cv2.resize(img_warp_colored.copy(), (img.shape[1], img.shape[0]))
    numbers_painted_show = cv2.resize(numbers_painted.copy(), (img.shape[1], img.shape[0]))
    solved_painted_show = cv2.resize(solved_painted.copy(), (img.shape[1], img.shape[0]))

    ### Calculado a matrix usada para voltar pra perspectiva do sudoku
    back_matrix = cv2.getPerspectiveTransform(pts2, pts1)

    ### Colocando o resultado numa imagem preta com a mesma perspectiva do sudoku
    img_warp_blank = np.zeros((img_warp_colored.shape[0], img_warp_colored.shape[1], 3), np.uint8)
    img_warp_blank = paint_image(solved_numbers, img_warp_blank, (0, 255, 0))
    back_img_warp_blank = cv2.warpPerspective(img_warp_blank.copy(), back_matrix, (img.shape[1], img.shape[0]))

    ### Colocando o resultado na imagem original (tipo chroma key)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 0, 0])
    mask = cv2.inRange(back_img_warp_blank.copy(), lower_black, upper_black)

    masked_image = back_img_warp_blank.copy()
    masked_image[mask != 0] = [0, 0, 0]

    masked_backgroud = img.copy()
    masked_backgroud[mask == 0] = [0, 0, 0]

    img_solved = masked_image + masked_backgroud

    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(2,5)

    axarr[0][0].imshow(cv2.cvtColor(img, cv2.cv2.COLOR_BGR2RGB))
    axarr[0][0].title.set_text('Original')
    axarr[0][1].imshow(cv2.cvtColor(img_gray, cv2.cv2.COLOR_BGR2RGB))
    axarr[0][1].title.set_text('Escala de cinza')
    axarr[0][2].imshow(cv2.cvtColor(img_blur, cv2.cv2.COLOR_BGR2RGB))
    axarr[0][2].title.set_text('Gaussian Blur')
    axarr[0][3].imshow(cv2.cvtColor(img_threshold, cv2.cv2.COLOR_BGR2RGB))
    axarr[0][3].title.set_text('Adaptive Threshold')
    axarr[0][4].imshow(cv2.cvtColor(img_contours, cv2.cv2.COLOR_BGR2RGB))
    axarr[0][4].title.set_text('Contornos')
    axarr[1][0].imshow(cv2.cvtColor(img_bigcontour, cv2.cv2.COLOR_BGR2RGB))
    axarr[1][0].title.set_text('Maior Contorno')
    axarr[1][1].imshow(cv2.cvtColor(numbers_painted_show, cv2.cv2.COLOR_BGR2RGB))
    axarr[1][1].title.set_text('Números preditos')
    axarr[1][2].imshow(cv2.cvtColor(solved_painted_show, cv2.cv2.COLOR_BGR2RGB))
    axarr[1][2].title.set_text('Números da Solução')
    axarr[1][3].imshow(cv2.cvtColor(back_img_warp_blank, cv2.cv2.COLOR_BGR2RGB))
    axarr[1][3].title.set_text('Perpectiva original')
    axarr[1][4].imshow(cv2.cvtColor(img_solved, cv2.cv2.COLOR_BGR2RGB))
    axarr[1][4].title.set_text('Imagem Original Resolvida')

    for i in range(2):
        for j in range(5):
            axarr[i][j].axis('off')

    plt.subplots_adjust(left=0.01, bottom=0, right=0.99, top=1, wspace=0.05, hspace=0.1)

    plt.show()
else:
    print('Não foi encontrado um contorno !')