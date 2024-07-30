import os
import numpy as np
import json
import cv2 # OpenCV para processamento dos vídeos 

# Ver doc. MediaPipe Holistic (OLD): https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Para desenhar a saída em um quadro: https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from google.protobuf import timestamp_pb2

def mp_draw_image(rgb_image, landmarks_result):
    """Função para desenhar a saída do MediaPipe em um quadro.

    Args:
        rgb_image : TYPE
            Quadro da imagem no formato de RGB (usar OpenCV2). \n
        landmarks_result : TYPE
            Resultado dos pontos-chave após processamento do MediaPipe Pose. \n

    Returns:
        annotated_image : TYPE
            Quadro contendo os pontos-chave do MediaPipe anotados. \n
    """
    pose_landmarks_list = landmarks_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Percorrendo cada ponto-chave calculado
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Desenhando os pontos sobre o quadro
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(  # type: ignore
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(  # type: ignore
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,  # type: ignore
            solutions.drawing_styles.get_default_pose_landmarks_style(),  # type: ignore
        )

    return annotated_image


def mp_process_video(video_path: str, video_name: str, output_path: str):
    """Função para Aplicar o MediaPipe Holistic em UM vídeo.

    Args:
        video_path : str
            Caminho do vídeo (com extensão) para processamento (ex: C:/video.mp4). \n
        video_name : str
            Nome do arquivo de vídeo para criação dos vídeos de pós-processamento (ex: video.mp4). \n
        output_path : str
            Pasta de saída para criação dos vídeos de pós-processamento (ex: C:/). \n

    Returns:
        video_landmarks : list ['Quadro1': [landmark0, landmark1, ...], ...]
            Lista contendo os pontos rastreados, por quadro, de um vídeo. \n
    """

    # Definindo o modelo a ser utilizado
    # actual_dir = os.path.dirname(__file__)
    holistic_models_path = os.path.join('./', "holistic_models") # Pasta contendo os modelos do MediaPipe
    holistic_models = {
        'Lite': os.path.join(holistic_models_path, "pose_landmarker_lite.task"),
        'Full': os.path.join(holistic_models_path, "pose_landmarker_full.task"),
        'Heavy': os.path.join(holistic_models_path, "pose_landmarker_heavy.task"),
    }
    model = holistic_models.get('Lite') # Selecionar um modelo para o Holistic

    # Criando a tarefa do MediaPipe (configuração do grafo de funcionamento)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Criando a tarefa de rastreamento de pose no MODO DE VÍDEO
    # Ver parâmetros em: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python#configuration_options
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    # Ininicializando o grafo de funcionamento do MediaPipe Holistic
    with PoseLandmarker.create_from_options(options) as landmarker:

        # Carregando o vídeo com o OpenCV
        video = cv2.VideoCapture(video_path)
        frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)  # Largura do Vídeo
        frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Altura do Vídeo

        print("Video Resolution: ", frame_width, " x ", frame_height)
        print("Video FPS: ", video.get(cv2.CAP_PROP_FPS))
        print("Video Frame Count:", video.get(cv2.CAP_PROP_FRAME_COUNT))

        #! Criação de um vídeo de saída, com o resultado de processamento
        video_output_test = cv2.VideoWriter(
            filename=os.path.join(output_path, f'{video_name}_output.mp4'),
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            # fourcc=cv2.VideoWriter.fourcc(*"MP4V"),
            fps=video.get(cv2.CAP_PROP_FPS),
            frameSize=(int(frame_width), int(frame_height)),
        )

        # Percorrendo os quadros do vídeo
        frames_landmarks = []
        frame_count = 0
        while video.isOpened():
            sucess, frame = video.read()
            if not sucess:
                print("Ignorando quadro vazio")
                break #'break' para vídeo e 'continue' para live-vídeo

            # Convertendo o quadro obtido para um objeto válido do MediaPipe (BGR para RGB)
            frame_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_converted
            )

            # Aplicando a tarefa de rastreamento do Holistic/Pose Estimator
            frame_timestamp = video.get(cv2.CAP_PROP_POS_MSEC) # Tempo em segundos
            # print('Processando: ', mp_image, " ", frame_timestamp)
            pose_landmarker_result = landmarker.detect_for_video(
                mp_image, 
                mp.Timestamp.from_seconds(frame_timestamp).microseconds() # Tempo em Microsegundos
            )

            # Percorrendo cada ponto-chave encontrado no quadro para salvar em uma lista
            formatted_pose_landmarks = []
            for idx in range(len(pose_landmarker_result.pose_landmarks)):
                pose_landmarks = pose_landmarker_result.pose_landmarks[idx]

                # Salvando os pontos, formatadamente, para a lista de saída
                for num, landmark in enumerate(pose_landmarks):
                    formatted_pose_landmarks.append(
                        {
                            f"ponto_{num}": {
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                            }
                        }
                    )

            # Salvando os pontos, formatadamente, para a lista de saída
            frames_landmarks.append({f"quadro_{frame_count}": formatted_pose_landmarks})
            frame_count = frame_count + 1

            #! Criando o quadro para vídeo de saída (BGR)
            annotated_frame = np.full(
                (int(frame_height), int(frame_width), 3), 0, dtype=np.uint8
            )  # Apenas os pontos com fundo personalizável
            # annotated_frame = frame.copy() # Colocar os pontos em cima do quadro do vídeo original
            annotated_frame = mp_draw_image(annotated_frame, pose_landmarker_result)
            video_output_test.write(annotated_frame)

    # Fechando os streams dos vídeos de entrada e saída
    video.release()
    video_output_test.release()

    return frames_landmarks


def process_folder_videos(input_folder: str, output_folder: str):
    """Processar os vídeos das subpastas de uma pasta, sendo essas
    subpastas com o nome da classe dos vídeos (ex: para uso com V-LIBRASIL).

    Args:
        input_folder : str
            Caminho da pasta que contém as subpastas para processamento. \n
        output_folder : str
            Caminho da pasta para saída dos arquivos JSON do processamento. \n
    
    Notes:
        Os arquivos JSON de saída são organizados por classe
    
    """
    # Percorrer todas as subpastas (classes) na pasta de entrada
    #for class_name in os.listdir(input_folder):
    #   class_path = os.path.join(input_folder, class_name)
#
        # Para cada classe, processar os vídeos internos
 #       if os.path.isdir(class_path):
            # Diretório para saída do processo, POR CLASSE
        #class_output_folder = os.path.join(output_folder, class_name)
    os.makedirs(output_folder, exist_ok=True)

    # Percorrer cada vídeo e processar com MediaPipe
    # json_output_list = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            # Configurando o caminho do vídeo, com extensão
            video_path = os.path.join(input_folder, filename)

            # Procesando com MediaPipe Holistic
            print(f"Processando... -> '{filename}'")
            video_landmarks = mp_process_video(
                video_path, os.path.splitext(filename)[0], output_folder
            )
            print(f"Processado! -> '{filename}'")

            # Salvando os resultados do vídeo para uma lista formatada
            json_output_list = [
                {
                    "nome_video": f"{filename}",
                    #"classe:": f"{class_name}",
                    "landmarks_quadros": video_landmarks,
                }
            ]

            # Salvando as variáveis para saída no JSON
            output_path = os.path.join(
                output_folder,
                f"{filename}_landmarks.json",
            )

            # Salvar os landmarks POR CLASSE no arquivo JSON
            with open(output_path, "w") as outfile:
                json.dump(json_output_list, outfile, indent=4)


def main():
    """Função principal do programa
    """

    # Caminhos para a pasta contendo os vídeos de entrada e saída para o JSON
    input_folder = "./videos UFPE (V-LIBRASIL)/data"  #! Cada classe de um dataset teria que ser atribuída nesta varíavel
    output_folder = "./saida_processamento"  #! O(s) JSON(s) de saída ficam nas subpastas (classes) desta variável

    # Processar todos os vídeos da pasta de entrada e salvar o JSON resultante na pasta de saída
    process_folder_videos(input_folder=input_folder, output_folder=output_folder)


if __name__ == '__main__':
    """Ponto de entrada do programa
    """

    #Chama a função principal
    main()
