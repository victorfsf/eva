from dateutil import parser as dateutil_parser
from functools import partial

__all__ = ['date_parse']


class PortugueseParserInfo(dateutil_parser.parserinfo):

    HMS = [
        ('h', 'hr', 'hrs', 'hora', 'horas'),
        ('m', 'min', 'minuto', 'minutos'),
        ('s', 'seg', 'segundo', 'segundos')
    ]
    JUMP = [
        ' ', '.', ',', ';', '-', '/', "'",
        'às', 'em', 'e', 'de', 'º'
    ]
    MONTHS = [
        ('jan', 'janeiro'), ('fev', 'fevereiro'), ('mar', 'março', 'marco'),
        ('abr', 'abril'), ('mai', 'maio'), ('jun', 'junho'),
        ('jul', 'julho'), ('ago', 'agosto'),
        ('set', 'setembro'), ('out', 'outubro'),
        ('nov', 'novembro'), ('dez', 'dezembro')
    ]
    PERTAIN = ['de']
    WEEKDAYS = [
        ('seg', 'segunda', 'segunda-feira', 'segunda feira'),
        ('ter', 'terça', 'terca', 'terça-feira',
         'terca-feira', 'terça feira', 'terca feira'),
        ('qua', 'quarta', 'quarta-feira', 'quarta feira'),
        ('qui', 'quinta', 'quinta-feira', 'quinta feira'),
        ('sex', 'sexta', 'sexta-feira', 'sexta feira'),
        ('sab', 'sábado', 'sabado'), ('dom', 'domingo')
    ]


date_parse = partial(
    dateutil_parser.parse,
    fuzzy=True,
    parserinfo=PortugueseParserInfo(dayfirst=True)
)
