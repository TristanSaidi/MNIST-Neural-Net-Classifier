import pygame
import sys
import numpy as np
import model
layers_dims = [784, 200, 100, 10]
layer_activation_list = ["tanh","tanh","softmax"]

class pixel:

    def __init__(self, rect):
        self.rect = rect
        self.color = (0,0,0)
        self.clicked = False
        self.peripheral_clicked = False

    def click(self, surface):
        self.clicked = True
        surface.fill((0,0,0), self.rect)

    def peripheral_click(self, surface):
        if not self.clicked:
            self.peripheral_clicked = True
            surface.fill((10,10,10), self.rect) #grey

    def draw_pixel(self, surface):
        pygame.draw.rect(surface, self.color , self.rect, 1)

def initialize_grid(x_start, y_start, x_width, y_width, surface):

    pixel_list = []

    for i in range(28):
        for j in range(28):
            p = pixel(pygame.Rect(x_start+j*(x_width-1),y_start+i*(y_width-1), x_width, y_width))
            pixel_list.append(p)
            p.draw_pixel(surface)
    return pixel_list

def vectorize_input(pixel_list):

    x = np.zeros((784,1))

    for i in range(784):
        if pixel_list[i].clicked == True:
            x[i][0] = 1*255
        if pixel_list[i].peripheral_clicked == True:
            x[i][0] = 0.5*255
    return x

def predict(pixel_list, learned_parameters):

    x = vectorize_input(pixel_list)
    print("Model predicts this is a: " + str(model.model_predict(x, learned_parameters)[0]))

def initialize_display(learned_parameters):

    DISPLAY_WIDTH = 500
    DISPLAY_HEIGHT = 600
    PIXEL_WIDTH = 15
    PIXEL_HEIGHT = 15

    surface = pygame.display.set_mode(size = (DISPLAY_WIDTH,DISPLAY_HEIGHT))
    surface.fill((255,255,255))



    pixel_list = initialize_grid((DISPLAY_WIDTH-((PIXEL_WIDTH-1)*28))//2, 20, PIXEL_WIDTH, PIXEL_HEIGHT,surface)

    mouse_drag = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_drag = True
                position = pygame.mouse.get_pos()
                clicked_pixel = [p for p in pixel_list if p.rect.collidepoint(position)]
                for pix in clicked_pixel:
                    pix.click(surface)
                    try:
                        pixel_list[pixel_list.index(pix)+1].peripheral_click(surface)
                    except:
                        pass
                    try:
                        if(pixel_list.index(pix) > 1):
                            pixel_list[pixel_list.index(pix)-1].peripheral_click(surface)
                    except:
                        pass
                    try:
                        pixel_list[pixel_list.index(pix)+28].peripheral_click(surface)
                    except:
                        pass
                    try:
                        if(pixel_list.index(pix) > 28):
                            pixel_list[pixel_list.index(pix)-28].peripheral_click(surface)
                    except:
                        pass
                    pygame.display.update(pix.rect)

            if event.type == pygame.MOUSEBUTTONUP:
                mouse_drag = False

            if event.type == pygame.MOUSEMOTION:
                if mouse_drag:
                    position = pygame.mouse.get_pos()
                    clicked_pixel = [p for p in pixel_list if p.rect.collidepoint(position)]
                    for pix in clicked_pixel:
                        pix.click(surface)
                        try:
                            pixel_list[pixel_list.index(pix)+1].peripheral_click(surface)
                        except:
                            pass
                        try:
                            if(pixel_list.index(pix) > 1):
                                pixel_list[pixel_list.index(pix)-1].peripheral_click(surface)
                        except:
                            pass
                        try:
                            pixel_list[pixel_list.index(pix)+28].peripheral_click(surface)
                        except:
                            pass
                        try:
                            if(pixel_list.index(pix) > 28):
                                pixel_list[pixel_list.index(pix)-28].peripheral_click(surface)
                        except:
                            pass
                        pygame.display.update(pix.rect)

            if event.type == pygame.KEYDOWN:
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_r]:
                    initialize_display(learned_parameters) # resets
                elif pressed[pygame.K_RETURN]:
                    predict(pixel_list, learned_parameters)
                    initialize_display(learned_parameters) # resets
        pygame.display.update()

if __name__ == '__main__':
    x_train = np.loadtxt('train_X.csv', delimiter = ',').T
    y_train = np.loadtxt('train_label.csv', delimiter = ',').T

    x_test = np.loadtxt('test_X.csv', delimiter = ',').T
    y_test = np.loadtxt('test_label.csv', delimiter = ',').T

    learned_parameters, costs = model.generate_model(x_train, y_train, layers_dims, layer_activation_list, 0.08, 150)
    model.predict(x_test, y_test, learned_parameters)
    initialize_display(learned_parameters)
