#IA
import openai
import faiss
import numpy as np
import pickle
import requests
from dotenv import load_dotenv
import os

# CONFIGURACOES INICIAIS DA IA
CHAVE = os.getenv("OPENAI_API_KEY")
openai.api_key = CHAVE  # Substitua pela sua chave

# CARREGA A MEMÓRIA DA IA
index = faiss.read_index("meu_indice.faiss")
with open("blocos.pkl", "rb") as f:
    blocos = pickle.load(f)

# IA
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

# CONFIGURACOES DO META API
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")  # use esse mesmo no painel da Meta

def responder_whatsapp(NUMBER, MENSAGEM):
    ACCESS_TOKEN = os.getenv("WHATSAPP_TOKEN")  # Seu token da Cloud API
    PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")  # ID do número do WhatsApp Cloud
    DESTINATARIO = NUMBER  # Número do cliente no formato E.164

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

def transcrever_audio(caminho_audio):
    with open(caminho_audio, "rb") as audio_file:
        resposta = openai.Audio.transcribe("whisper-1", audio_file)
    return resposta["text"]

def baixar_audio(media_id, salvar_em="audio.ogg"):
    ACCESS_TOKEN = os.getenv("WHATSAPP_TOKEN")  # Seu token da Cloud API

    # 1. Pegar a URL da mídia
    url_info = f"https://graph.facebook.com/v18.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    resposta = requests.get(url_info, headers=headers)
    media_url = resposta.json().get("url")

    # 2. Baixar o arquivo de mídia
    if media_url:
        resposta_arquivo = requests.get(media_url, headers=headers)
        with open(salvar_em, "wb") as f:
            f.write(resposta_arquivo.content)
        return salvar_em
    else:
        raise Exception("Não foi possível obter a URL do áudio.")

#API
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
    corpo = await request.json()

    try:
        mensagem = corpo['entry'][0]['changes'][0]['value']['messages'][0]
        tipo = mensagem['type']
        numero = mensagem['from']

        if tipo == "text":
            texto = mensagem['text']['body']
            resposta = ia(texto)
        elif tipo == "audio":
            media_id = mensagem['audio']['id']
            caminho = baixar_audio(media_id)
            transcricao = transcrever_audio(caminho)
            resposta = ia(transcricao)
        else:
            resposta = "Desculpe, não entendi."

        return responder_whatsapp(numero, resposta)

    except Exception as e:
        print("Erro:", e)
        return {"erro": str(e)}