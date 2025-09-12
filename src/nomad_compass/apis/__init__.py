from nomad.config.models.plugins import APIEntryPoint


class MyAPIEntryPoint(APIEntryPoint):

    def load(self):
        from nomad_compass.apis.myapi import app

        return app


myapi = MyAPIEntryPoint(
    prefix = 'myapi',
    name = 'MyAPI',
    description = 'My custom API.',
)