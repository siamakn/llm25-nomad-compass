from fastapi import FastAPI
from nomad.config import config

myapi_entry_point = config.get_plugin_entry_point('nomad_compass.apis:myapi')

app = FastAPI(
    root_path=f'{config.services.api_base_path}/{myapi_entry_point.prefix}'
)

@app.get('/')
async def root():
    return {"message": "Hello World"}