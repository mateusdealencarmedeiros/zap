#IA
import openai
import faiss
import numpy as np
import pickle
import requests
from dotenv import load_dotenv
import os

# CONFIGURACOES

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")  # use esse mesmo no painel da Meta
CHAVE = os.getenv("OPENAI_API_KEY")
openai.api_key = CHAVE  # Substitua pela sua chave

# CARREGA A MEMÓRIA DA IA

index = faiss.read_index("meu_indice.faiss")
with open("blocos.pkl", "rb") as f:
    blocos = pickle.load(f)

# FUNCOES

def transcrever_audio(media_id):
    ACCESS_TOKEN = os.getenv("WHATSAPP_TOKEN")

    # 1. Obter URL do arquivo de mídia
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    r = requests.get(url, headers=headers)
    media_url = r.json()["url"]

    # 2. Baixar o áudio
    audio_bytes = requests.get(media_url, headers=headers).content
    with open("audio.ogg", "wb") as f:
        f.write(audio_bytes)

    # 3. Enviar para Whisper (transcrição)
    with open("audio.ogg", "rb") as audio_file:
        resposta = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

    return resposta.strip()

def ia(pergunta):
    # Criando o embedding: representação númerica da pergunta
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta
    ).data[0].embedding

    # Busca os blocos mais próximos no índice
    D, I = index.search(np.array([emb]).astype("float32"), k=3)
    contexto = "\n\n".join([blocos[i] for i in I[0]])

    # Garante que o contexto total não exceda o tamanho máximo permitido
    if len(contexto) > 15000:
        contexto = contexto[:15000]

    # Envia para o ChatGPT com contexto restrito
    mensagens = [
        {"role": "system", "content": (
            "Você se chama Juliana, tem 34 anos, é casada e mãe de uma filha. Valoriza profundamente a família e leva essa essência para o seu trabalho. Atua como assistente virtual e especialista imobiliária na UNICO., empresa na qual encontrou propósito após uma experiência traumática vivida anos atrás, na compra do seu primeiro imóvel. Desde então, assumiu como missão pessoal ajudar outras pessoas a viverem esse processo com clareza, segurança e confiança."
            "Seu foco é garantir que nenhuma dúvida do cliente fique sem resposta. Você orienta com empatia e inteligência sobre tudo que envolve a jornada de aquisição: desde informações sobre os produtos e suas localizações, até valores, condições de pagamento, tabela de vendas e possíveis custos envolvidos. Sua postura é sempre solícita, segura e acolhedora, transmitindo ao cliente a certeza de que está sendo bem atendido por alguém que realmente entende do assunto."
            "Lembre-se: você é especialista, confiável e estratégica. Seu tom é sempre profissional, claro, respeitoso e assertivo.\n\n"
            f"Contexto:\n{contexto}"
        )},
        {"role": "user", "content": pergunta}
    ]

    resposta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=mensagens,
        temperature=0
    )
    return resposta.choices[0].message.content.strip()

def responder_whatsapp(NUMBER, MENSAGEM, TIPO):
    ACCESS_TOKEN = os.getenv("WHATSAPP_TOKEN")  # Seu token da Cloud API
    PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")  # ID do número do WhatsApp Cloud
    DESTINATARIO = NUMBER  # Número do cliente no formato E.164
    
    if TIPO == 'text':
        # === ENVIO ===
        url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": DESTINATARIO,
            "type": "text",
            "text": {"body": MENSAGEM}
        }

        response = requests.post(url, headers=headers, json=data)

        return response.status_code
    elif TIPO == 'audio':
        # === Passo 1: Upload do áudio ===
        url_upload = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/media"
        headers_upload = {
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }
        files = {
            "file": (os.path.basename('resposta.mp3'), open('resposta.mp3', 'rb')),
            "messaging_product": (None, "whatsapp"),
            "type": (None, "audio/mpeg")  # use "audio/ogg" se for .ogg
        }

        r_upload = requests.post(url_upload, headers=headers_upload, files=files)
        print("Upload status:", r_upload.status_code, r_upload.text)

        if r_upload.status_code != 200:
            return {"erro": "Erro ao fazer upload do áudio"}

        media_id = r_upload.json()["id"]

        # === Passo 2: Enviar a mensagem de áudio ===
        url_send = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
        headers_send = {
            "Authorization": f"Bearer {ACCESS_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": NUMBER,
            "type": "audio",
            "audio": {
                "id": media_id
            }
        }

        r_send = requests.post(url_send, headers=headers_send, json=payload)
        print("Envio status:", r_send.status_code, r_send.text)
        return r_send.status_code

# API

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Verificação inicial da URL (GET)
@app.get("/webhook")
def verificar_webhook(request: Request):
    args = request.query_params
    if args.get("hub.mode") == "subscribe" and args.get("hub.verify_token") == VERIFY_TOKEN:
        return PlainTextResponse(args.get("hub.challenge"))
    return PlainTextResponse("Erro de verificação", status_code=403)

# Receber mensagens (POST)
@app.post("/webhook")
async def receber_mensagem(request: Request):
    corpo = await request.json()
    print(corpo)

    try:
        mensagem = corpo['entry'][0]['changes'][0]['value']['messages'][0]
        tipo = mensagem['type']
        numero = mensagem['from']

        if tipo == "text":
            texto = mensagem['text']['body']
        elif tipo == "audio":
            media_id = mensagem['audio']['id']
            texto = transcrever_audio(media_id)
        else:
            texto = f"Mensagem do tipo '{tipo}' ainda não é suportada."

        print(numero)
        print(texto)

        resposta = ia(texto)
        return responder_whatsapp(numero, resposta, tipo)

    except Exception as e:
        print("Erro:", e)
        return {"erro": str(e)}