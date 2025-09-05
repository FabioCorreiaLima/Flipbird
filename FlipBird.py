import pygame
import random
import sys
import os
import neat

# ----------------------------- Configurações da Tela ----------------------------- #
TELA_LARGURA = 900
TELA_ALTURA = 800

IMAGEM_CANO = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'pipe.png')))
IMAGEM_CHAO = pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'base.png')))
IMAGEM_BACKGROUND = pygame.transform.scale(
    pygame.image.load(os.path.join('imgs', 'bg.png')), 
    (TELA_LARGURA, TELA_ALTURA)
)
IMAGEMS_PASSARO = [
    pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird1.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird2.png'))),
    pygame.transform.scale2x(pygame.image.load(os.path.join('imgs', 'bird3.png')))
]

pygame.font.init()
FONTE_PONTUACAO = pygame.font.SysFont('arial', 50)

# ----------------------------- Classes do Jogo ----------------------------- #
class Passaro:
    IMGS = IMAGEMS_PASSARO
    ROTACAO_MAXIMA = 25
    VELOCIDADE_ROTACAO = 20
    TEMPO_ANIMACAO = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angulo = 0
        self.velocidade = 0
        self.altura = y
        self.tempo = 0
        self.contagem_imagem = 0
        self.imagem = self.IMGS[0]

    def mover(self):
        self.tempo += 1
        deslocamento = self.velocidade * self.tempo + 1.5 * self.tempo**2
        if deslocamento > 16:
            deslocamento = 16
        elif deslocamento < 0:
            deslocamento -= 2
        self.y += deslocamento

        if deslocamento < 0 or self.y < self.altura + 50:
            if self.angulo < self.ROTACAO_MAXIMA:
                self.angulo = self.ROTACAO_MAXIMA
        else:
            if self.angulo > -90:
                self.angulo -= self.VELOCIDADE_ROTACAO

    def pular(self):
        self.velocidade = -10.5
        self.tempo = 0
        self.altura = self.y

    def desenhar(self, tela):
        self.contagem_imagem += 1
        if self.contagem_imagem < self.TEMPO_ANIMACAO:
            self.imagem = self.IMGS[0]
        elif self.contagem_imagem < self.TEMPO_ANIMACAO*2:
            self.imagem = self.IMGS[1]
        elif self.contagem_imagem < self.TEMPO_ANIMACAO*3:
            self.imagem = self.IMGS[2]
        elif self.contagem_imagem < self.TEMPO_ANIMACAO*4:
            self.imagem = self.IMGS[1]
        elif self.contagem_imagem == self.TEMPO_ANIMACAO*4 + 1:
            self.imagem = self.IMGS[0]
            self.contagem_imagem = 0

        if self.angulo <= -80:
            self.imagem = self.IMGS[1]
            self.contagem_imagem = self.TEMPO_ANIMACAO*2

        imagem_rotacionada = pygame.transform.rotate(self.imagem, self.angulo)
        pos_centro = self.imagem.get_rect(topleft=(self.x, self.y)).center
        retangulo = imagem_rotacionada.get_rect(center=pos_centro)
        tela.blit(imagem_rotacionada, retangulo.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.imagem)

class Cano:
    IMAGEM = IMAGEM_CANO
    DISTANCIA = 200
    VELOCIDADE = 5

    def __init__(self, x):
        self.x = x
        self.altura = 0
        self.pos_topo = 0
        self.pos_base = 0
        self.CANO_TOPO = pygame.transform.flip(self.IMAGEM, False, True)
        self.CANO_BASE = self.IMAGEM
        self.passou = False
        self.definir_altura()

    def definir_altura(self):
        self.altura = random.randrange(50, 450)
        self.pos_topo = self.altura - self.CANO_TOPO.get_height()
        self.pos_base = self.altura + self.DISTANCIA

    def mover(self):
        self.x -= self.VELOCIDADE

    def desenhar(self, tela):
        tela.blit(self.CANO_TOPO, (self.x, self.pos_topo))
        tela.blit(self.CANO_BASE, (self.x, self.pos_base))

    def colisao(self, passaro):
        passaro_mask = passaro.get_mask()
        topo_mask = pygame.mask.from_surface(self.CANO_TOPO)
        base_mask = pygame.mask.from_surface(self.CANO_BASE)

        distancia_topo = (self.x - passaro.x, self.pos_topo - round(passaro.y))
        distancia_base = (self.x - passaro.x, self.pos_base - round(passaro.y))

        if passaro_mask.overlap(topo_mask, distancia_topo) or passaro_mask.overlap(base_mask, distancia_base):
            return True
        return False

class Chao:
    IMAGEM = IMAGEM_CHAO
    VELOCIDADE = 5
    LARGURA = IMAGEM_CHAO.get_width()

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.LARGURA

    def mover(self):
        self.x1 -= self.VELOCIDADE
        self.x2 -= self.VELOCIDADE

        if self.x1 + self.LARGURA <= 0:
            self.x1 = self.x2 + self.LARGURA
        if self.x2 + self.LARGURA <= 0:
            self.x2 = self.x1 + self.LARGURA

    def desenhar(self, tela):
        tela.blit(self.IMAGEM, (self.x1, self.y))
        tela.blit(self.IMAGEM, (self.x2, self.y))

# ----------------------------- Funções ----------------------------- #
def desenhar_tela(tela, passaros, canos, chao, pontos):
    tela.blit(IMAGEM_BACKGROUND, (0,0))
    for passaro in passaros:
        passaro.desenhar(tela)
    for cano in canos:
        cano.desenhar(tela)
    texto = FONTE_PONTUACAO.render(f"Pontuação: {pontos}", 1, (255,255,255))
    tela.blit(texto, (TELA_LARGURA - 10 - texto.get_width(), 10))
    chao.desenhar(tela)
    pygame.display.update()

def jogar_genoma(genoma, config):
    net = neat.nn.FeedForwardNetwork.create(genoma, config)
    passaros = [Passaro(230, 350)]
    canos = [Cano(600)]
    chao = Chao(700)
    pontos = 0
    rodando = True
    clock = pygame.time.Clock()

    while rodando:
        clock.tick(30)
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                rodando = False
                pygame.quit()
                sys.exit()

        for i, passaro in enumerate(passaros):
            passaro.mover()
            if len(canos) > 0:
                cano = canos[0]
                distancia_x = cano.x - passaro.x
                entrada = (passaro.y, distancia_x, cano.altura)
                output = net.activate(entrada)
                if output[0] > 0.5:
                    passaro.pular()

        adicionar_cano = False
        remover_canos = []
        for cano in canos:
            for i, passaro in enumerate(passaros):
                if cano.colisao(passaro):
                    passaros.pop(i)
                if not cano.passou and passaro.x > cano.x + cano.CANO_TOPO.get_width():
                    cano.passou = True
                    adicionar_cano = True
            cano.mover()
            if cano.x + cano.CANO_TOPO.get_width() < 0:
                remover_canos.append(cano)

        if adicionar_cano:
            pontos += 1
            canos.append(Cano(600))

        for cano in remover_canos:
            canos.remove(cano)

        for i, passaro in enumerate(passaros):
            if passaro.y + passaro.imagem.get_height() >= 700 or passaro.y < 0:
                passaros.pop(i)

        if len(passaros) == 0:
            rodando = False

    return pontos

def fitness_genomas(genomas, config):
    for _, genoma in genomas:
        genoma.fitness = jogar_genoma(genoma, config)

def executar_neat(caminho_config):
    caminho_config = os.path.abspath(caminho_config)
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        caminho_config
    )

    populacao = neat.Population(config)
    populacao.add_reporter(neat.StdOutReporter(True))
    populacao.add_reporter(neat.StatisticsReporter())

    winner = populacao.run(fitness_genomas, 50)
    print("Melhor genoma:", winner)

# ----------------------------- Main ----------------------------- #
if __name__ == "__main__":
    caminho_config = os.path.join(os.getcwd(), "config.txt")
    executar_neat(caminho_config)
