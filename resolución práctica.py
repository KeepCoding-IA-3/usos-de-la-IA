from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import os
import json

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o"
)

parser = StrOutputParser()

# Paso 1: Extraer información del email
extract_info_prompt = PromptTemplate(
    input_variables=["email"],
    template="""
A partir del siguiente email, extrae un diccionario JSON con la información clave:

- pedido_id: número de pedido
- cliente: nombre del remitente
- contacto: email o forma de contacto
- motivo: motivo de la solicitud
- detalles: detalles relevantes para decidir si se acepta o no la devolución

Email:
{email}

Responde exclusivamente con un JSON válido y limpio, sin usar etiquetas como ```json ni comillas triples.
"""
)

chain_extraccion = LLMChain(
    llm=llm,
    prompt=extract_info_prompt,
    output_key="text",
    output_parser=parser
)

# Paso 2: Evaluar decisión
decision_prompt = PromptTemplate(
    input_variables=["motivo", "detalles"],
    template="""
Analiza el motivo y los detalles de una solicitud de devolución para decidir si se acepta o no.

Aceptar si:
- Defecto de fabricación
- Error en el suministro
- Producto incompleto de fábrica

Rechazar si:
- Daños en transporte no asegurado
- Uso indebido del cliente
- Solicitud fuera de plazo

Devuelve un JSON válido y limpio con:
- aceptar: true o false
- razon: explicación breve

No incluyas etiquetas como ```json ni comillas triples.
Motivo: {motivo}
Detalles: {detalles}
"""
)

chain_decision = LLMChain(
    llm=llm,
    prompt=decision_prompt,
    output_key="text",
    output_parser=parser
)

# Paso 3: Redactar la respuesta
respuesta_prompt = PromptTemplate(
    input_variables=["cliente", "pedido_id", "aceptar", "razon"],
    template="""
Redacta un email formal para el cliente {cliente}, sobre el pedido {pedido_id}. 

Indica si se acepta o no la solicitud de devolución, justifica la decisión con la siguiente razón:
{razon}

Sé cordial y profesional. Firma como: "Atención al cliente de CII".

Devuelve solo el contenido del email, sin etiquetas de formato ni explicaciones.
"""
)

chain_respuesta = LLMChain(
    llm=llm,
    prompt=respuesta_prompt,
    output_key="text",
    output_parser=parser
)

# --- Email de ejemplo ---
email_text = (
    "Asunto: Solicitud de reemplazo por daños en transporte – Pedido #D347-STELLA\n"
    "Estimado equipo de Componentes Intergalácticos Industriales S.A.,\n"
    "Me pongo en contacto con ustedes como cliente reciente para comunicar una incidencia relacionada con "
    "el pedido #D347-STELLA, correspondiente a un lote de condensadores de fluzo modelo FX-88, destinados "
    "a un proyecto estratégico de gran envergadura: la construcción de la Estrella de la Muerte.\n"
    "Lamentablemente, al recibir el envío, observamos que varios de los condensadores presentaban daños "
    "visibles y no funcionales debido a una caída durante el transporte interestelar. Dado que estos "
    "componentes son críticos para el núcleo central del sistema de rayos destructores, solicitamos el reemplazo "
    "inmediato de las unidades defectuosas y una revisión de los protocolos de embalaje y transporte.\n"
    "Contacto: dmarquez@imperiumgalactic.net"
)

# --- Ejecución paso a paso con JSON directo ---

# Paso 1: Extraer información
info_json = chain_extraccion.invoke({"email": email_text})["text"]
info = json.loads(info_json)

# Paso 2: Evaluar decisión
decision_json = chain_decision.invoke({
    "motivo": info["motivo"],
    "detalles": info["detalles"]
})["text"]
decision = json.loads(decision_json)

# Paso 3: Redactar respuesta
respuesta = chain_respuesta.invoke({
    "cliente": info["cliente"],
    "pedido_id": info["pedido_id"],
    "aceptar": str(decision["aceptar"]).lower(),
    "razon": decision["razon"]
})["text"]

# Mostrar resultado final
print(respuesta)