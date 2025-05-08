from  openai import OpenAI

client = OpenAI(api_key="tu clave de open ai")
response = client.responses.create(
    model="gpt-4o",
    input="Como de guapos son los alumnos de keepcoding?"

)
print(response.output_text)
response = client.responses.create(
    model="gpt-4o",
    input=f"di todo lo contrarioa esto {response.output_text}"

)
print(response.output_text)