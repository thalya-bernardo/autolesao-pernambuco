# ============================================================
# DASHBOARD — AUTOLESÃO EM ADOLESCENTES EM PERNAMBUCO
# SINAN | 10–19 ANOS | 2014–2024
# ============================================================

import os
import re
import zipfile
import tempfile
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Autolesão em Adolescentes em Pernambuco",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1550px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: Arial, sans-serif;
    }

    .titulo-principal {
        text-align: center;
        font-size: 31px;
        font-weight: 800;
        margin-bottom: 2px;
    }

    .subtitulo {
        text-align: center;
        font-size: 16px;
        margin-bottom: 22px;
    }

    .titulo-secao {
        font-size: 19px;
        font-weight: 800;
        margin-top: 25px;
        margin-bottom: 12px;
        border-bottom: 1px solid #ddd;
        padding-bottom: 7px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #e4e4e4;
        border-radius: 12px;
        padding: 14px;
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 29px;
        font-weight: 700;
    }

    .nota {
        font-size: 12px;
        color: #666;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3. ARQUIVOS ORIGINAIS
# ============================================================

ARQUIVO_SINAN = "BD_AUTOLESÃO_COMPLETO (1).xlsx"
ARQUIVO_POP = "dados_pernambuco_combinados (1).xlsx"
ARQUIVO_MALHA = "PE_Municipios_2025.zip"

# ============================================================
# 4. FUNÇÕES AUXILIARES
# ============================================================

def idade_em_anos(valor):

    if pd.isna(valor):
        return np.nan

    try:
        valor = int(float(valor))
    except:
        return np.nan

    unidade = valor // 1000
    numero = valor % 1000

    if unidade == 4:
        return numero

    return np.nan


def codigo_municipio_6d(valor):

    if pd.isna(valor):
        return pd.NA

    try:
        return str(int(float(valor))).zfill(6)[:6]
    except:
        texto = re.sub(r"\D", "", str(valor))
        return texto[:6] if len(texto) >= 6 else pd.NA


def decodificar_sexo(valor):

    mapa = {
        "M": "Masculino",
        "F": "Feminino",
        "I": "Ignorado",
        1: "Masculino",
        2: "Feminino",
        9: "Ignorado"
    }

    if pd.isna(valor):
        return "Não informado"

    if valor in mapa:
        return mapa[valor]

    try:
        return mapa.get(int(float(valor)), "Não informado")
    except:
        return mapa.get(
            str(valor).strip().upper(),
            "Não informado"
        )


def decodificar_raca(valor):

    mapa = {
        1: "Branca",
        2: "Preta",
        3: "Amarela",
        4: "Parda",
        5: "Indígena",
        9: "Ignorado"
    }

    if pd.isna(valor):
        return "Não informado"

    try:
        return mapa.get(
            int(float(valor)),
            "Não informado"
        )
    except:
        return "Não informado"


def decodificar_recorrencia(valor):

    mapa = {
        1: "Sim",
        2: "Não",
        9: "Ignorado"
    }

    if pd.isna(valor):
        return "Não informado"

    try:
        return mapa.get(
            int(float(valor)),
            "Não informado"
        )
    except:
        return "Não informado"


def numero_br(valor, casas=0):

    if pd.isna(valor):
        return "—"

    if casas == 0:
        return f"{valor:,.0f}".replace(",", ".")

    texto = f"{valor:,.{casas}f}"
    texto = texto.replace(",", "X").replace(".", ",").replace("X", ".")

    return texto


# ============================================================
# 5. MÉTODOS
# ============================================================

METODOS = {
    "AG_FORCA": "Força corporal/espancamento",
    "AG_ENFOR": "Enforcamento",
    "AG_OBJETO": "Objeto contundente",
    "AG_CORTE": "Objeto perfurocortante",
    "AG_QUENTE": "Substância/objeto quente",
    "AG_ENVEN": "Envenenamento/intoxicação",
    "AG_FOGO": "Arma de fogo",
    "AG_AMEACA": "Ameaça",
    "AG_OUTROS": "Outros"
}

# ============================================================
# 6. CARREGAMENTO E PREPARAÇÃO
# ============================================================

@st.cache_data(show_spinner=False)
def carregar_dados():

    # --------------------------------------------------------
    # SINAN
    # --------------------------------------------------------

    sinan = pd.read_excel(
        ARQUIVO_SINAN,
        sheet_name="AUTOLESAO_PE",
        engine="openpyxl"
    )

    sinan["IDADE_ANOS"] = sinan["NU_IDADE_N"].apply(
        idade_em_anos
    )

    sinan["NU_ANO"] = pd.to_numeric(
        sinan["NU_ANO"],
        errors="coerce"
    )

    sinan["LES_AUTOP"] = pd.to_numeric(
        sinan["LES_AUTOP"],
        errors="coerce"
    )

    # Universo da pesquisa
    sinan = sinan[
        (sinan["NU_ANO"].between(2014, 2024)) &
        (sinan["IDADE_ANOS"].between(10, 19)) &
        (sinan["LES_AUTOP"] == 1)
    ].copy()

    # Validação
    if len(sinan) != 12713:
        raise ValueError(
            f"A base deveria reproduzir 12.713 registros, "
            f"mas foram encontrados {len(sinan)}."
        )

    # --------------------------------------------------------
    # Variáveis derivadas
    # --------------------------------------------------------

    sinan["FAIXA_ETARIA"] = pd.cut(
        sinan["IDADE_ANOS"],
        bins=[9, 14, 19],
        labels=["10 a 14 anos", "15 a 19 anos"]
    )

    sinan["SEXO_DESC"] = sinan["CS_SEXO"].apply(
        decodificar_sexo
    )

    sinan["RACA_COR_DESC"] = sinan["CS_RACA"].apply(
        decodificar_raca
    )

    sinan["RECORRENCIA_DESC"] = sinan["OUT_VEZES"].apply(
        decodificar_recorrencia
    )

    sinan["COD_MUN_6"] = sinan["ID_MN_RESI"].apply(
        codigo_municipio_6d
    )

    # Métodos
    for campo in METODOS:

        if campo in sinan.columns:
            sinan[campo] = pd.to_numeric(
                sinan[campo],
                errors="coerce"
            )

    # --------------------------------------------------------
    # POPULAÇÃO
    # --------------------------------------------------------

    pop = pd.read_excel(
        ARQUIVO_POP,
        sheet_name="consolidado",
        dtype=str,
        engine="openpyxl"
    )

    pop["COD_MUN"] = (
        pop["COD_MUN"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    pop["COD_MUN_6"] = pop["COD_MUN"].str[:6]

    pop["ANO"] = pd.to_numeric(
        pop["ANO"],
        errors="coerce"
    )

    pop["IDADE"] = pd.to_numeric(
        pop["IDADE"],
        errors="coerce"
    )

    pop["SEXO"] = pd.to_numeric(
        pop["SEXO"],
        errors="coerce"
    )

    pop["POPULAÇÃO"] = (
        pop["POPULAÇÃO"]
        .astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    pop["POPULAÇÃO"] = pd.to_numeric(
        pop["POPULAÇÃO"],
        errors="coerce"
    )

    pop = pop[
        (pop["ANO"].between(2014, 2024)) &
        (pop["IDADE"].between(10, 19))
    ].copy()

    # --------------------------------------------------------
    # Município e GERES
    # --------------------------------------------------------

    municipios = (
        pop[
            [
                "COD_MUN_6",
                "MUNICÍPIO",
                "REGIÃO DE SAÚDE"
            ]
        ]
        .drop_duplicates("COD_MUN_6")
    )

    sinan = sinan.merge(
        municipios,
        on="COD_MUN_6",
        how="left"
    )

    return sinan, pop


@st.cache_data(show_spinner=False)
def carregar_malha():

    pasta = tempfile.mkdtemp()

    with zipfile.ZipFile(
        ARQUIVO_MALHA,
        "r"
    ) as arquivo_zip:

        arquivo_zip.extractall(pasta)

    shp = [
        os.path.join(pasta, arquivo)
        for arquivo in os.listdir(pasta)
        if arquivo.lower().endswith(".shp")
    ][0]

    geo = gpd.read_file(shp)

    geo["COD_MUN_6"] = (
        geo["CD_MUN"]
        .astype(str)
        .str[:6]
    )

    geo = geo.to_crs(epsg=4326)

    return geo


# ============================================================
# 7. CARREGAR
# ============================================================

try:

    with st.spinner("Carregando bancos originais..."):

        df, pop = carregar_dados()
        geo = carregar_malha()

except Exception as erro:

    st.error(
        "Não foi possível carregar os bancos originais."
    )

    st.exception(erro)
    st.stop()


# ============================================================
# 8. CABEÇALHO
# ============================================================

st.markdown(
    """
    <div class="titulo-principal">
        AUTOLESÃO EM ADOLESCENTES EM PERNAMBUCO
    </div>

    <div class="subtitulo">
        Notificações registradas no SINAN |
        10 a 19 anos |
        2014–2024
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 9. FILTROS GLOBAIS
# ============================================================

st.markdown(
    '<div class="titulo-secao">FILTROS</div>',
    unsafe_allow_html=True
)

f1, f2, f3, f4 = st.columns(4)

with f1:

    anos = st.multiselect(
        "Ano",
        sorted(df["NU_ANO"].dropna().astype(int).unique()),
        placeholder="Todos"
    )

with f2:

    regioes = st.multiselect(
        "Região de Saúde",
        sorted(df["REGIÃO DE SAÚDE"].dropna().unique()),
        placeholder="Todas"
    )

with f3:

    municipios = st.multiselect(
        "Município",
        sorted(df["MUNICÍPIO"].dropna().unique()),
        placeholder="Todos"
    )

with f4:

    sexos = st.multiselect(
        "Sexo",
        ["Feminino", "Masculino"],
        placeholder="Todos"
    )


f5, f6, f7, f8 = st.columns(4)

with f5:

    faixas = st.multiselect(
        "Faixa etária",
        ["10 a 14 anos", "15 a 19 anos"],
        placeholder="Todas"
    )

with f6:

    racas = st.multiselect(
        "Raça/cor",
        [
            "Branca",
            "Preta",
            "Amarela",
            "Parda",
            "Indígena"
        ],
        placeholder="Todas"
    )

with f7:

    recorrencias = st.multiselect(
        "Recorrência",
        ["Sim", "Não"],
        placeholder="Todas"
    )

with f8:

    metodos_selecionados = st.multiselect(
        "Método utilizado",
        list(METODOS.values()),
        placeholder="Todos"
    )


# ============================================================
# 10. APLICAR FILTROS À BASE SINAN
# ============================================================

dados = df.copy()

if anos:
    dados = dados[
        dados["NU_ANO"].isin(anos)
    ]

if regioes:
    dados = dados[
        dados["REGIÃO DE SAÚDE"].isin(regioes)
    ]

if municipios:
    dados = dados[
        dados["MUNICÍPIO"].isin(municipios)
    ]

if sexos:
    dados = dados[
        dados["SEXO_DESC"].isin(sexos)
    ]

if faixas:
    dados = dados[
        dados["FAIXA_ETARIA"].isin(faixas)
    ]

if racas:
    dados = dados[
        dados["RACA_COR_DESC"].isin(racas)
    ]

if recorrencias:
    dados = dados[
        dados["RECORRENCIA_DESC"].isin(recorrencias)
    ]


# ============================================================
# 11. FILTRO DE MÉTODO
# ============================================================

if metodos_selecionados:

    campos = [
        campo
        for campo, descricao in METODOS.items()
        if descricao in metodos_selecionados
    ]

    mascara = pd.Series(
        False,
        index=dados.index
    )

    for campo in campos:
        mascara = mascara | (dados[campo] == 1)

    dados = dados[mascara]


# ============================================================
# 12. POPULAÇÃO CORRESPONDENTE
# ============================================================

pop_filtrada = pop.copy()

if anos:
    pop_filtrada = pop_filtrada[
        pop_filtrada["ANO"].isin(anos)
    ]

if municipios:

    codigos = df[
        df["MUNICÍPIO"].isin(municipios)
    ]["COD_MUN_6"].unique()

    pop_filtrada = pop_filtrada[
        pop_filtrada["COD_MUN_6"].isin(codigos)
    ]

elif regioes:

    codigos = df[
        df["REGIÃO DE SAÚDE"].isin(regioes)
    ]["COD_MUN_6"].unique()

    pop_filtrada = pop_filtrada[
        pop_filtrada["COD_MUN_6"].isin(codigos)
    ]


# Sexo populacional
# Base original:
# 1 e 2.
# Confirmamos na auditoria a estrutura.
if sexos:

    cod_sexo = []

    if "Masculino" in sexos:
        cod_sexo.append(1)

    if "Feminino" in sexos:
        cod_sexo.append(2)

    pop_filtrada = pop_filtrada[
        pop_filtrada["SEXO"].isin(cod_sexo)
    ]


# Faixa etária populacional
if faixas:

    idades = []

    if "10 a 14 anos" in faixas:
        idades.extend(range(10, 15))

    if "15 a 19 anos" in faixas:
        idades.extend(range(15, 20))

    pop_filtrada = pop_filtrada[
        pop_filtrada["IDADE"].isin(idades)
    ]


# ============================================================
# 13. KPIs
# ============================================================

total = len(dados)

# Taxa só é calculada quando os filtros possuem
# denominadores populacionais compatíveis.

filtro_sem_denominador = (
    bool(racas) or
    bool(recorrencias) or
    bool(metodos_selecionados)
)

if not filtro_sem_denominador:

    populacao_total = pop_filtrada["POPULAÇÃO"].sum()

    taxa = (
        total / populacao_total * 100000
        if populacao_total > 0
        else np.nan
    )

else:
    taxa = np.nan


# Feminino entre sexo válido
sexo_valido = dados[
    dados["SEXO_DESC"].isin(
        ["Feminino", "Masculino"]
    )
]

if len(sexo_valido) > 0:

    pct_feminino = (
        (sexo_valido["SEXO_DESC"] == "Feminino")
        .mean()
        * 100
    )

else:
    pct_feminino = np.nan


# Recorrência entre respostas válidas
rec_valida = dados[
    dados["RECORRENCIA_DESC"].isin(
        ["Sim", "Não"]
    )
]

if len(rec_valida) > 0:

    pct_recorrencia = (
        (rec_valida["RECORRENCIA_DESC"] == "Sim")
        .mean()
        * 100
    )

else:
    pct_recorrencia = np.nan


# ============================================================
# 14. VISÃO GERAL
# ============================================================

st.markdown(
    '<div class="titulo-secao">VISÃO GERAL</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Total de notificações",
    numero_br(total)
)

if pd.notna(taxa):

    k2.metric(
        "Taxa por 100 mil adolescentes",
        numero_br(taxa, 1)
    )

else:

    k2.metric(
        "Taxa por 100 mil adolescentes",
        "—"
    )


k3.metric(
    "Sexo feminino",
    (
        f"{numero_br(pct_feminino, 2)}%"
        if pd.notna(pct_feminino)
        else "—"
    )
)

k4.metric(
    "Recorrência",
    (
        f"{numero_br(pct_recorrencia, 2)}%"
        if pd.notna(pct_recorrencia)
        else "—"
    )
)

if filtro_sem_denominador:

    st.caption(
        "A taxa populacional não é recalculada quando estão "
        "ativos filtros de raça/cor, recorrência ou método, "
        "pois a base populacional utilizada não possui "
        "denominadores correspondentes para essas dimensões."
    )


# ============================================================
# 15. EVOLUÇÃO TEMPORAL
# ============================================================

st.markdown(
    '<div class="titulo-secao">EVOLUÇÃO TEMPORAL</div>',
    unsafe_allow_html=True
)

serie = (
    dados.groupby("NU_ANO")
    .size()
    .reset_index(name="Notificações")
)

fig_serie = px.line(
    serie,
    x="NU_ANO",
    y="Notificações",
    markers=True,
    title="Número de notificações por ano"
)

fig_serie.update_layout(
    xaxis_title="Ano",
    yaxis_title="Notificações",
    hovermode="x unified"
)

st.plotly_chart(
    fig_serie,
    use_container_width=True
)


# ============================================================
# 16. TAXA ANUAL 2014–2024
# ============================================================

# Somente filtros com denominador disponível
if not filtro_sem_denominador:

    pop_ano = (
        pop_filtrada
        .groupby("ANO", as_index=False)["POPULAÇÃO"]
        .sum()
    )

    taxa_ano = serie.merge(
        pop_ano,
        left_on="NU_ANO",
        right_on="ANO",
        how="left"
    )

    taxa_ano["Taxa por 100 mil"] = (
        taxa_ano["Notificações"] /
        taxa_ano["POPULAÇÃO"] *
        100000
    )

    fig_taxa = px.line(
        taxa_ano,
        x="NU_ANO",
        y="Taxa por 100 mil",
        markers=True,
        title=(
            "Taxa de notificações de autolesão "
            "por 100 mil adolescentes"
        )
    )

    fig_taxa.update_layout(
        xaxis_title="Ano",
        yaxis_title="Taxa por 100 mil",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_taxa,
        use_container_width=True
    )


# ============================================================
# 17. RAÇA/COR × ANO
# ============================================================

dados_raca = dados[
    dados["RACA_COR_DESC"].isin(
        [
            "Branca",
            "Preta",
            "Amarela",
            "Parda",
            "Indígena"
        ]
    )
]

raca_ano = (
    dados_raca
    .groupby(
        ["NU_ANO", "RACA_COR_DESC"]
    )
    .size()
    .reset_index(name="Notificações")
)

fig_raca_ano = px.line(
    raca_ano,
    x="NU_ANO",
    y="Notificações",
    color="RACA_COR_DESC",
    markers=True,
    title=(
        "Notificações de autolesão por raça/cor "
        "e ano de notificação"
    )
)

fig_raca_ano.update_layout(
    xaxis_title="Ano",
    yaxis_title="Notificações",
    legend_title="Raça/cor"
)

st.plotly_chart(
    fig_raca_ano,
    use_container_width=True
)


# ============================================================
# 18. DISTRIBUIÇÃO TERRITORIAL
# ============================================================

st.markdown(
    '<div class="titulo-secao">DISTRIBUIÇÃO TERRITORIAL</div>',
    unsafe_allow_html=True
)

municipal = (
    dados.groupby(
        ["COD_MUN_6", "MUNICÍPIO"],
        dropna=False
    )
    .size()
    .reset_index(name="Notificações")
)


# População municipal
pop_municipal = (
    pop_filtrada
    .groupby(
        "COD_MUN_6",
        as_index=False
    )["POPULAÇÃO"]
    .sum()
)

municipal = municipal.merge(
    pop_municipal,
    on="COD_MUN_6",
    how="left"
)

municipal["Taxa por 100 mil"] = (
    municipal["Notificações"] /
    municipal["POPULAÇÃO"] *
    100000
)


mapa = geo.merge(
    municipal,
    on="COD_MUN_6",
    how="left"
)

mapa["Notificações"] = (
    mapa["Notificações"]
    .fillna(0)
)


territorio1, territorio2 = st.columns(
    [1.25, 0.75]
)

with territorio1:

    fig_mapa = px.choropleth(
        mapa,
        geojson=mapa.geometry.__geo_interface__,
        locations=mapa.index,
        color="Notificações",
        hover_name="NM_MUN",
        hover_data={
            "Notificações": True
        },
        title="Notificações por município de residência"
    )

    fig_mapa.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig_mapa.update_layout(
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0
        )
    )

    st.plotly_chart(
        fig_mapa,
        use_container_width=True
    )


with territorio2:

    ranking = (
        municipal
        .sort_values(
            "Notificações",
            ascending=False
        )
        .head(10)
        .sort_values("Notificações")
    )

    fig_ranking = px.bar(
        ranking,
        x="Notificações",
        y="MUNICÍPIO",
        orientation="h",
        title="10 municípios com mais notificações"
    )

    fig_ranking.update_layout(
        xaxis_title="Notificações",
        yaxis_title=""
    )

    st.plotly_chart(
        fig_ranking,
        use_container_width=True
    )


# ============================================================
# 19. REGIÕES DE SAÚDE
# ============================================================

regiao = (
    dados.groupby("REGIÃO DE SAÚDE")
    .size()
    .reset_index(name="Notificações")
    .sort_values("Notificações")
)

fig_regiao = px.bar(
    regiao,
    x="Notificações",
    y="REGIÃO DE SAÚDE",
    orientation="h",
    title="Distribuição das notificações por Região de Saúde"
)

fig_regiao.update_layout(
    xaxis_title="Notificações",
    yaxis_title=""
)

st.plotly_chart(
    fig_regiao,
    use_container_width=True
)


# ============================================================
# 20. PERFIL DAS NOTIFICAÇÕES
# ============================================================

st.markdown(
    '<div class="titulo-secao">PERFIL DAS NOTIFICAÇÕES</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Sexo
# ------------------------------------------------------------

perfil1, perfil2 = st.columns(2)

with perfil1:

    sexo_plot = (
        dados[
            dados["SEXO_DESC"].isin(
                ["Feminino", "Masculino"]
            )
        ]
        ["SEXO_DESC"]
        .value_counts()
        .reset_index()
    )

    sexo_plot.columns = [
        "Sexo",
        "Notificações"
    ]

    fig_sexo = px.pie(
        sexo_plot,
        names="Sexo",
        values="Notificações",
        hole=0.45,
        title="Sexo"
    )

    fig_sexo.update_traces(
        textinfo="percent+value"
    )

    st.plotly_chart(
        fig_sexo,
        use_container_width=True
    )


with perfil2:

    faixa_plot = (
        dados["FAIXA_ETARIA"]
        .value_counts(sort=False)
        .reset_index()
    )

    faixa_plot.columns = [
        "Faixa etária",
        "Notificações"
    ]

    fig_faixa = px.pie(
        faixa_plot,
        names="Faixa etária",
        values="Notificações",
        hole=0.45,
        title="Faixa etária"
    )

    fig_faixa.update_traces(
        textinfo="percent+value"
    )

    st.plotly_chart(
        fig_faixa,
        use_container_width=True
    )


# ------------------------------------------------------------
# Raça e recorrência
# ------------------------------------------------------------

perfil3, perfil4 = st.columns(2)

with perfil3:

    raca_plot = (
        dados[
            dados["RACA_COR_DESC"].isin(
                [
                    "Branca",
                    "Preta",
                    "Amarela",
                    "Parda",
                    "Indígena"
                ]
            )
        ]
        ["RACA_COR_DESC"]
        .value_counts()
        .reset_index()
    )

    raca_plot.columns = [
        "Raça/cor",
        "Notificações"
    ]

    fig_raca = px.bar(
        raca_plot,
        x="Raça/cor",
        y="Notificações",
        title="Raça/cor",
        text="Notificações"
    )

    st.plotly_chart(
        fig_raca,
        use_container_width=True
    )


with perfil4:

    rec_plot = (
        dados[
            dados["RECORRENCIA_DESC"].isin(
                ["Sim", "Não"]
            )
        ]
        ["RECORRENCIA_DESC"]
        .value_counts()
        .reset_index()
    )

    rec_plot.columns = [
        "Recorrência",
        "Notificações"
    ]

    fig_rec = px.pie(
        rec_plot,
        names="Recorrência",
        values="Notificações",
        hole=0.45,
        title="Ocorrência anterior / recorrência"
    )

    fig_rec.update_traces(
        textinfo="percent+value"
    )

    st.plotly_chart(
        fig_rec,
        use_container_width=True
    )

    st.caption(
        "Percentuais calculados somente entre respostas válidas "
        "(Sim/Não)."
    )


# ============================================================
# 21. MÉTODOS UTILIZADOS
# ============================================================

st.markdown(
    '<div class="titulo-secao">CARACTERÍSTICAS DA AUTOLESÃO</div>',
    unsafe_allow_html=True
)

# Formato longo
lista = []

for campo, descricao in METODOS.items():

    temp = dados[
        dados[campo] == 1
    ].copy()

    if len(temp) == 0:
        continue

    temp["METODO"] = descricao

    lista.append(temp)


if lista:

    metodos_long = pd.concat(
        lista,
        ignore_index=True
    )

else:

    metodos_long = pd.DataFrame()


if len(metodos_long) > 0:

    dist_metodo = (
        metodos_long["METODO"]
        .value_counts()
        .reset_index()
    )

    dist_metodo.columns = [
        "Método",
        "Marcações"
    ]

    dist_metodo = dist_metodo.sort_values(
        "Marcações"
    )

    fig_metodo = px.bar(
        dist_metodo,
        x="Marcações",
        y="Método",
        orientation="h",
        text="Marcações",
        title="Métodos utilizados"
    )

    fig_metodo.update_layout(
        xaxis_title="Número de marcações",
        yaxis_title=""
    )

    st.plotly_chart(
        fig_metodo,
        use_container_width=True
    )

    st.caption(
        "Método é uma variável de resposta múltipla. "
        "Uma mesma notificação pode apresentar mais de um "
        "método registrado; portanto, as marcações podem "
        "superar o número total de notificações."
    )


# ============================================================
# 22. ANÁLISES CRUZADAS
# ============================================================

st.markdown(
    '<div class="titulo-secao">ANÁLISES CRUZADAS</div>',
    unsafe_allow_html=True
)

a1, a2 = st.columns(2)

with a1:

    variavel_cruzamento = st.selectbox(
        "Cruzar método com:",
        [
            "Sexo",
            "Raça/cor",
            "Faixa etária",
            "Recorrência",
            "Região de Saúde"
        ]
    )

with a2:

    tipo_medida = st.radio(
        "Exibir:",
        ["Número", "Percentual"],
        horizontal=True
    )


mapa_variavel = {
    "Sexo": "SEXO_DESC",
    "Raça/cor": "RACA_COR_DESC",
    "Faixa etária": "FAIXA_ETARIA",
    "Recorrência": "RECORRENCIA_DESC",
    "Região de Saúde": "REGIÃO DE SAÚDE"
}

coluna_cruzamento = mapa_variavel[
    variavel_cruzamento
]


if len(metodos_long) > 0:

    cruzamento = (
        metodos_long
        .groupby(
            ["METODO", coluna_cruzamento],
            dropna=False
        )
        .size()
        .reset_index(name="Número")
    )

    if tipo_medida == "Percentual":

        cruzamento["Valor"] = (
            cruzamento.groupby("METODO")["Número"]
            .transform(
                lambda x: x / x.sum() * 100
            )
        )

        eixo_x = "Valor"
        titulo_eixo = "Percentual (%)"

    else:

        cruzamento["Valor"] = cruzamento["Número"]

        eixo_x = "Valor"
        titulo_eixo = "Número de marcações"


    fig_cruzamento = px.bar(
        cruzamento,
        x=eixo_x,
        y="METODO",
        color=coluna_cruzamento,
        orientation="h",
        barmode="stack",
        title=(
            f"Método utilizado segundo "
            f"{variavel_cruzamento.lower()}"
        )
    )

    fig_cruzamento.update_layout(
        xaxis_title=titulo_eixo,
        yaxis_title="",
        legend_title=variavel_cruzamento
    )

    st.plotly_chart(
        fig_cruzamento,
        use_container_width=True
    )


# ============================================================
# 23. TABELA RESUMO
# ============================================================

st.markdown(
    '<div class="titulo-secao">RESUMO DOS DADOS FILTRADOS</div>',
    unsafe_allow_html=True
)

tabela_resumo = pd.DataFrame({
    "Indicador": [
        "Notificações",
        "Sexo feminino",
        "Sexo masculino",
        "10 a 14 anos",
        "15 a 19 anos",
        "Recorrência - Sim",
        "Recorrência - Não"
    ],
    "N": [
        len(dados),

        (dados["SEXO_DESC"] == "Feminino").sum(),

        (dados["SEXO_DESC"] == "Masculino").sum(),

        (
            dados["FAIXA_ETARIA"]
            == "10 a 14 anos"
        ).sum(),

        (
            dados["FAIXA_ETARIA"]
            == "15 a 19 anos"
        ).sum(),

        (
            dados["RECORRENCIA_DESC"]
            == "Sim"
        ).sum(),

        (
            dados["RECORRENCIA_DESC"]
            == "Não"
        ).sum()
    ]
})

st.dataframe(
    tabela_resumo,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 24. NOTAS METODOLÓGICAS
# ============================================================

with st.expander("Notas metodológicas"):

    st.markdown(
        """
        **População do estudo:** notificações de lesão
        autoprovocada entre adolescentes de 10 a 19 anos,
        residentes em Pernambuco, entre 2014 e 2024.

        **Fonte das notificações:** Sistema de Informação de
        Agravos de Notificação (SINAN).

        **Denominadores populacionais:** base populacional
        utilizada no estudo.

        **Taxas:** número de notificações dividido pela
        população adolescente correspondente, multiplicado
        por 100.000.

        **Recorrência:** calculada entre registros com resposta
        válida "Sim" ou "Não". Registros ignorados ou não
        informados são preservados no banco, mas excluídos do
        denominador dessa proporção.

        **Raça/cor:** registros ignorados e não informados são
        preservados para avaliação da completude, mas excluídos
        das distribuições epidemiológicas principais.

        **Método utilizado:** variável de resposta múltipla.
        Uma notificação pode possuir mais de um método marcado.
        Portanto, a soma das frequências dos métodos pode ser
        superior ao número de notificações.
        """
    )


# ============================================================
# 25. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "Fonte: SINAN e base populacional utilizada no estudo. "
    "Elaboração própria."
)
