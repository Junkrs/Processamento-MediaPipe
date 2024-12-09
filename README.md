# Processamento - MediaPipe ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
Parte do Trabalho de Conclusão de Curso desenvolvido para minha graduação em Ciência da Computação na [UNESP](https://www.fc.unesp.br/#!/departamentos/computacao/cursos-de-graduao/bacharelado-em-ciencia-da-computacao/).
> [!IMPORTANT]
> Este repositório corresponde ao processamento dos vídeos dos sinais de LIBRAS para geração dos arquivos JSON. Para a renderização e animação dos esqueletos dos avatares, veja o conteúdo disponibilizado [neste repositório](https://github.com/Junkrs/Ambiente---TCC). :leftwards_arrow_with_hook:

## 🔵 Introdução ao Projeto

Este trabalho tem como objetivo a criação de um sistema que capture e redirecione os movimentos da Língua Brasileira de Sinais (LIBRAS) para animar esqueletos de avatares 3D, usando ferramentas de inteligência artificial e visão computacional. Utilizando o [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) para rastreamento de pontos chaves em vídeos que contenham sinais de LIBRAS, foram animados esqueletos de avatares 3D dentro do Unity.

A monografia do projeto completo está disponível no [repositório institucional da UNESP](https://hdl.handle.net/11449/258220).

## 🔵 Como Usar

Existem 3 pastas importantes e 1 arquivo .py, que é responsável por todo o processamento.

- Pastas:
1. _Holistic_Models_: Essa pasta contém os modelos do MediaPipe. Neste caso, temos 3 versão do MediaPipe Pose, que foi utilizado no projeto.
2. _saida_processamento_: Essa pasta é o endereço de saída dos vídeos processados e também dos arquivos JSON gerados com cada vídeo.
3. _videos UFPE (V-LIBRASIL)_: Já esta pasta é onde estarão os vídeos originais para o processamento. Ela está escrita dessa forma pois o _dataset_ escolhido no TCC foi o [V-LIBRASIL](https://libras.cin.ufpe.br/).
<br>

- Código fonte:
  O código fonte é divido em funções que irão anotar, fazendo a ilustração dos _landmarks_ e depois irão gerar o arquivo JSON, com as coordenadas de cada _landmark_ em cada _frame_ do vídeo. Alguns dos trechos de código que são importantes para futuras modificações são:
  
  1. Modelo escolhido do MediaPipe (linha 80):
     
     ```
      model = holistic_models.get('Full')
     ```
  2.  Pontos do modelo escolhido do MediaPipe (linha 145):
     
       ```
       point_names = {...}
       ```
  3. Pontos que devem ser ignorados (Ppcional) (Linha 183):
     
     ```
      ignored_points = {...}
     ```
  4. Formato de saída dos dados no arquivo JSON (Linha 196):
     
     ```
      formatted_pose_landmarks.append(
       {
         f"ponto_{num} - {point_name}": {
           "x": landmark.x,
           "y": landmark.y,
           "z": landmark.z,
         }
       }
     )
     ```
Uma vez processados, os arquivos JSON da pasta de saída podem ser inseridos na [segunda parte](https://github.com/Junkrs/Ambiente---TCC) deste projeto. Desde que o formato esteja adequeado.

#### Qualquer dúvida, fique a vontade para entrar em contato comigo :D
