"""Исходный код лабораторной работы Gaia DR3.

Номер каждого блока соответствует вызову _gaia_run в Lab01_Gaia.ipynb.
При запуске из ноутбука блоки используют его общие переменные.
"""

# %% [1]
# @title Библиотеки и оформление графиков

import base64, hashlib, io, json, math, time, unittest, zlib
import html as html_lib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, special
from IPython.display import HTML, display

OUT = Path('gaia_team_results')
OUT.mkdir(exist_ok=True)

SEED = 20261020
ALPHA = 0.05

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,

    # Космическая палитра
    'figure.facecolor': '#0b1020',
    'axes.facecolor': '#131b30',
    'savefig.facecolor': '#0b1020',
    'text.color': '#edf3ff',
    'axes.labelcolor': '#edf3ff',
    'axes.edgecolor': '#34415f',
    'xtick.color': '#edf3ff',
    'ytick.color': '#edf3ff',

    # Сетка и границы
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#34415f',
    'grid.alpha': 0.45,
    'grid.linestyle': '--',
})

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 140)

COLORS = {
    'G': '#7AA2F7',
    'BP−RP': '#7EE0C3',
    'Параллакс': '#F7768E',
    'mean': '#E0AF68',
    'median': '#BB9AF7',
    'bright': '#FF79C6'
}

# Ссылка на фон из GitHub
BACKGROUND_URL = (
    "https://raw.githubusercontent.com/sofia28s/statistics/"
    "refs/heads/main/background.png"
)


def cosmic_card(content, title=None):
    """Вывод информации в космической карточке."""

    heading_html = ""

    if title is not None:
        heading_html = f"""
        <h2 style="
            color:#c9b6ff;
            font-size:25px;
            margin:0 0 18px;
        ">{title}</h2>
        """

    html_code = f"""
    <div style="
        width:100%;
        box-sizing:border-box;
        padding:30px;
        margin-top:25px;
        margin-bottom:25px;
        border-radius:22px;
        background-image:
            linear-gradient(rgba(4,8,25,.20), rgba(4,8,25,.20)),
            url('{BACKGROUND_URL}');
        background-size:cover;
        background-position:center;
        color:#f4f5ff;
        font-family:Arial,sans-serif;
        line-height:1.85;
    ">
        <div style="
            padding:28px 32px;
            border-radius:17px;
            background:rgba(5,10,30,.94);
            border:1px solid rgba(180,195,255,.30);
        ">
            {heading_html}
            {content}
        </div>
    </div>
    """

    display(HTML(html_code))


def styled_html_table(df, title=None, digits=6):
    """Таблицы с тёмной шапкой, сиреневыми и белыми строками."""

    table_df = df.copy()

    for col in table_df.select_dtypes(include="number").columns:
        table_df[col] = table_df[col].round(digits)

    headers = "".join(
        f"""
        <th style="
            padding:14px;
            text-align:left;
            background:#111827;
            color:#F8FAFC;
            font-weight:bold;
            border-bottom:1px solid #374151;
        ">{html_lib.escape(str(col))}</th>
        """
        for col in table_df.columns
    )

    rows = ""

    for i, row in enumerate(table_df.itertuples(index=False, name=None)):
        background = "#F5F3FF" if i % 2 == 0 else "#FFFFFF"
        cells = ""

        for value in row:
            if pd.isna(value):
                value = "—"

            cells += f"""
            <td style="
                padding:12px;
                background:{background};
                color:#1F2937;
                border-bottom:1px solid #E5E7EB;
                text-align:left;
            ">{html_lib.escape(str(value))}</td>
            """

        rows += f"<tr>{cells}</tr>"

    heading = ""

    if title:
        heading = f"""
        <h3 style="
            color:#1F2937;
            font-family:Arial,sans-serif;
            font-size:18px;
            margin:0 0 14px;
        ">{html_lib.escape(title)}</h3>
        """

    html_code = f"""
    <div style="font-family:Arial,sans-serif; margin:16px 0;">
        {heading}
        <div style="overflow-x:auto; border-radius:10px;">
            <table style="
                width:100%;
                border-collapse:separate;
                border-spacing:0;
                border-radius:10px;
                overflow:hidden;
                border:1px solid #D8D4F0;
                font-size:14px;
                color:#1F2937;
            ">
                <thead>
                    <tr>{headers}</tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
    </div>
    """

    display(HTML(html_code))

# %% [2]
# @title

cosmic_card("""
<h1 style="
    color:#ffffff;
    font-size:30px;
    line-height:1.35;
    margin:0 0 8px;
">
    Лабораторная работа 01
</h1>

<div style="color:#b9a8ff; margin-bottom:24px;">
    — статистическое исследование объектов Gaia DR3
</div>

<div style="
    background:rgba(139,157,255,.09);
    border-left:3px solid #a5b4fc;
    padding:14px 18px;
    border-radius:0 8px 8px 0;
    margin-bottom:20px;
">
    <strong style="color:#c4b5fd;">Тема исследования</strong>
    <p style="margin:6px 0 0;">
        Как распределены видимая яркость, цвет и параллакс объектов Gaia
        и что меняется при выборе только самых ярких объектов.
    </p>
</div>

<p style="margin:0 0 26px;">
    <strong style="color:#c4b5fd;">Команда:</strong>
    три инфохимика · общая аналитическая часть для последующего оформления дизайнерами
</p>

<h3 style="
    color:#d8ddff;
    font-size:19px;
    margin:0 0 12px;
">
    Три исследуемые случайные величины
</h3>

<div style="
    display:flex;
    flex-wrap:wrap;
    gap:12px;
    margin-bottom:26px;
">

    <div style="
        flex:1 1 190px;
        background:rgba(105,125,220,.13);
        padding:15px;
        border-radius:10px;
        border:1px solid rgba(165,180,255,.22);
    ">
        <div style="color:#a5b4fc; font-size:13px;">Участник 1</div>
        <div style="font-weight:bold; font-size:19px; margin:5px 0;">G</div>
        <div style="font-size:13px; color:#cbd5f5;">
            Видимая звёздная величина
        </div>
    </div>

    <div style="
        flex:1 1 190px;
        background:rgba(105,125,220,.13);
        padding:15px;
        border-radius:10px;
        border:1px solid rgba(165,180,255,.22);
    ">
        <div style="color:#a5b4fc; font-size:13px;">Участник 2</div>
        <div style="font-weight:bold; font-size:19px; margin:5px 0;">BP − RP</div>
        <div style="font-size:13px; color:#cbd5f5;">
            Цвет объекта
        </div>
    </div>

    <div style="
        flex:1 1 190px;
        background:rgba(105,125,220,.13);
        padding:15px;
        border-radius:10px;
        border:1px solid rgba(165,180,255,.22);
    ">
        <div style="color:#a5b4fc; font-size:13px;">Участник 3</div>
        <div style="font-weight:bold; font-size:19px; margin:5px 0;">
            Параллакс
        </div>
        <div style="font-size:13px; color:#cbd5f5;">
            Измеренное угловое смещение
        </div>
    </div>

</div>

<div style="
    background:rgba(19,28,51,.92);
    border-left:3px solid #8b9dff;
    padding:18px 20px;
    border-radius:0 10px 10px 0;
    margin-bottom:18px;
">
    <div style="
        color:#c6d2ff;
        font-size:18px;
        font-weight:bold;
        margin-bottom:8px;
    ">
        Главный вопрос работы
    </div>
    <p style="margin:0;">
        Можно ли по случайной выборке объектов Gaia судить о всей выбранной базе
        и как меняется вывод, если выбирать объекты не случайно,
        а по яркости?
    </p>
</div>

<div style="
    background:rgba(23,27,50,.92);
    border:1px solid rgba(180,190,255,.2);
    padding:18px 20px;
    border-radius:10px;
">
    <div style="
        color:#c4b5fd;
        font-size:17px;
        font-weight:bold;
        margin-bottom:8px;
    ">
        Область выводов
    </div>
    <p style="margin:0;">
        Генеральная совокупность в этой лабораторной — конечная рабочая база
        из 10 000 объектов Gaia DR3. Это не все звёзды Вселенной и не все
        объекты Млечного Пути. Поэтому выводы относятся к этой базе
        и к способу её формирования.
    </p>
</div>
""", title=None)

# %% [3]
# @title

display(HTML(f"""
<div style="padding:35px; border-radius:18px; text-align:center;
background:linear-gradient(rgba(5,10,30,.25),rgba(5,10,30,.25)),url('{BACKGROUND_URL}') center/cover;">
<img src="https://raw.githubusercontent.com/sofia28s/statistics/refs/heads/main/musya.jpg"
style="width:250px;max-width:100%;border-radius:12px;">
</div>
"""))

# %% [4]
# @title
from IPython.display import display, HTML

display(HTML("""
<div style="
    background: #111827;
    color: #F8FAFC;
    padding: 20px 24px;
    border-left: 5px solid #A78BFA;
    border-radius: 10px;
    font-family: Arial, sans-serif;
    box-sizing: border-box;
">
    <h1 style="
        color: #F8FAFC;
        margin: 0;
        font-size: 28px;
        font-weight: 700;
        line-height: 1.5;
    ">
        Оглавление
    </h1>
</div>
"""))

# %% [5]
# @title
from IPython.display import display, HTML

display(HTML("""
<div id="data" style="
    background: #111827;
    color: #F8FAFC;
    padding: 20px 24px;
    border: 1px solid #475569;
    border-left: 5px solid #A78BFA;
    border-radius: 10px;
    font-family: Arial, sans-serif;
    box-sizing: border-box;
">
    <h2 style="
        color: #F8FAFC;
        margin: 0;
        font-size: 23px;
        font-weight: 700;
        line-height: 1.5;
    ">
        1. Данные и способ их получения
    </h2>
</div>
"""))

# %% [6]
# @title Источник и формирование данных

cosmic_card("""
<style>
.data-section {
    padding: 18px 22px;
    margin: 16px 0;
    border-left: 4px solid #A78BFA;
    border-radius: 10px;
    background: rgba(255,255,255,0.06);
}

.data-label {
    color: #C9B6FF;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}

.sql-window {
    margin: 18px 0;
    border: 1px solid rgba(180,195,255,0.25);
    border-radius: 12px;
    overflow: hidden;
    background: rgba(5,10,25,0.85);
}

.sql-top {
    padding: 11px 16px;
    background: rgba(255,255,255,0.07);
    color: #C9B6FF;
    font-size: 13px;
}

.sql-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 5px;
}

.sql-window pre {
    padding: 16px 20px;
    margin: 0;
    overflow-x: auto;
    color: #B9D7FF;
    font-size: 13px;
    line-height: 1.8;
}

.data-section p {
    margin: 0;
    color: #F4F5FF;
}
</style>

<div style="margin-bottom:24px;">
    <div style="color:#C9B6FF; font-size:12px; letter-spacing:3px; font-weight:bold;">
        DATA SOURCE · GAIA DR3
    </div>
    <div style="font-size:25px; font-weight:bold; margin-top:7px;">
        Архив астрономических наблюдений
    </div>
    <div style="margin-top:10px; color:#7EE0C3; font-size:12px; letter-spacing:1px;">
        ● OFFICIAL DATA
    </div>
</div>

<div class="data-section">
    <div class="data-label">01 / ИСТОЧНИК</div>
    <p>Источник данных — официальный архив Gaia Data Release 3, таблица gaiadr3.gaia_source.</p>
</div>

<div class="sql-window">
    <div class="sql-top">
        <span class="sql-dot" style="background:#F7768E;"></span>
        <span class="sql-dot" style="background:#E0AF68;"></span>
        <span class="sql-dot" style="background:#7EE0C3;"></span>
        &nbsp; gaia_query.sql
    </div>
    <pre><code>SELECT source_id, random_index, ra, dec, phot_g_mean_mag, bp_rp,
       parallax, parallax_error, ruwe, phot_g_mean_flux_over_error
FROM gaiadr3.gaia_source
WHERE random_index &lt; 20000
  AND phot_g_mean_mag IS NOT NULL
  AND bp_rp IS NOT NULL
  AND parallax IS NOT NULL
ORDER BY random_index</code></pre>
</div>

<div class="data-section" style="border-left-color:#7AA2F7;">
    <div class="data-label" style="color:#7AA2F7;">02 / ВЫБОРКА</div>
    <p>Для анализа используем 10 000 объектов с заполненными значениями G, BP − RP и параллакса. Объекты упорядочены по random_index — случайному индексу каталога. Поэтому первые строки в таком порядке позволяют получить случайный поднабор объектов, у которых доступны все три характеристики.</p>
</div>

<div class="data-section" style="border-left-color:#E0AF68;">
    <div class="data-label" style="color:#E0AF68;">03 / ОГРАНИЧЕНИЕ ОБЪЁМА</div>
    <p>Мы ограничили объём десятью тысячами строк, чтобы многократно повторять расчёты и сравнивать выборки разного размера без обработки всего каталога. Само число 10 000 не гарантирует, что наша база точно представляет все звёзды: на её состав влияют возможности наблюдений Gaia и требование наличия трёх измерений.</p>
</div>

<div class="data-section" style="border-left-color:#F7768E;">
    <div class="data-label" style="color:#F7768E;">04 / ПОДГОТОВЛЕННЫЙ ФАЙЛ</div>
    <p>При повторной проверке запрос выше вернул 14 932 подходящие записи. Однако в текущий ноутбук загружается уже подготовленный файл gaia_team.csv, содержащий 10 000 строк. Копия CSV в код ноутбука не встроена, поэтому для запуска нужен этот отдельный файл.</p>
</div>

<div class="data-section" style="border-left-color:#7EE0C3;">
    <div class="data-label" style="color:#7EE0C3;">05 / ГЕНЕРАЛЬНАЯ СОВОКУПНОСТЬ И ПАРАЛЛАКС</div>
    <p>Далее эти 10 000 строк считаем конечной генеральной совокупностью для наших вычислительных экспериментов. Из неё будем многократно выбирать меньшие группы и сравнивать их характеристики с известными характеристиками всей рабочей базы. Отрицательные измеренные параллаксы не удаляем. Из-за ошибок измерения оценка параллакса может оказаться отрицательной, но это не означает отрицательного физического расстояния. Удаление таких значений может смещать выборку, поэтому сохраняем их при анализе.</p>
</div>
""")

# %% [7]
# @title

columns_info = pd.DataFrame([
    {
        "Столбец": "phot_g_mean_mag",
        "Простое объяснение": "Видимая звёздная величина G; чем меньше число, тем ярче объект",
        "Единицы": "mag"
    },
    {
        "Столбец": "bp_rp",
        "Простое объяснение": "Разность синего и красного каналов; большее значение обычно означает более красный цвет",
        "Единицы": "mag"
    },
    {
        "Столбец": "parallax",
        "Простое объяснение": "Измеренное годовое угловое смещение объекта из-за движения Земли",
        "Единицы": "mas"
    },
    {
        "Столбец": "parallax_error",
        "Простое объяснение": "Стандартная ошибка измерения параллакса",
        "Единицы": "mas"
    },
    {
        "Столбец": "ruwe",
        "Простое объяснение": "Показатель качества астрометрического решения",
        "Единицы": "Без единицы"
    }
])

styled_html_table(
    columns_info,
    digits=3
)

# %% [8]
# @title Получение трех величин и условия наблюдений

cosmic_card("""
<style>
.gaia-intro {
    color:#DCE6FF;
    font-size:15px;
    margin-bottom:22px;
}
.gaia-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:12px;
    margin:20px 0 30px;
}
.gaia-item {
    padding:19px;
    border-radius:14px;
    background:linear-gradient(145deg,rgba(125,100,230,.17),rgba(65,150,230,.07));
    border:1px solid rgba(180,195,255,.25);
    border-top:3px solid var(--accent);
}
.gaia-item code {
    display:inline-block;
    color:var(--accent);
    font-size:14px;
    font-weight:bold;
    margin-bottom:12px;
}
.gaia-item p {
    margin:0;
    font-size:14px;
    line-height:1.7;
}
.gaia-heading {
    display:flex;
    align-items:center;
    gap:12px;
    margin:32px 0 18px;
}
.gaia-heading h2 {
    color:#C9B6FF;
    font-size:23px;
    margin:0;
}
.gaia-info {
    margin:13px 0;
    padding:19px 22px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.18);
    border-left:3px solid var(--accent);
    color:#F4F5FF;
    line-height:1.8;
}
.gaia-info p {
    margin:0;
}
.gaia-note {
    color:#AFC4E8;
    font-size:12px;
    letter-spacing:1.5px;
    margin:0 0 12px;
    font-weight:bold;
}
</style>

<h2 style="color:#C9B6FF;font-size:27px;margin:0 0 12px;">
    Получение трех величин
</h2>

<p class="gaia-intro">
    Все три характеристики взяты из таблицы
    <code style="color:#B9D9FF;">gaiadr3.gaia_source</code>, но получены они разными способами:
</p>

<div class="gaia-grid">
    <div class="gaia-item" style="--accent:#C9B6FF;">
        <code>phot_g_mean_mag</code>
        <p>— звёздная величина G, рассчитанная по среднему потоку света в широкой полосе G;</p>
    </div>

    <div class="gaia-item" style="--accent:#7AA2F7;">
        <code>bp_rp</code>
        <p>— разность звёздных величин в синем и красном фотометрических каналах Gaia;</p>
    </div>

    <div class="gaia-item" style="--accent:#7EE0C3;">
        <code>parallax</code>
        <p>— измеренный параллакс, полученный при обработке повторных наблюдений положения объекта.</p>
    </div>
</div>

<div style="height:1px;background:linear-gradient(90deg,transparent,#8B7BCB,transparent);margin:30px 0;"></div>

<div class="gaia-heading">
    <h2>Условия наблюдений и ограничения</h2>
</div>

<div class="gaia-info" style="--accent:#7AA2F7;">
    <div class="gaia-note">ПЕРИОД НАБЛЮДЕНИЙ</div>
    <p>
        Gaia DR3 использует наблюдения с 25 июля 2014 года по 28 мая 2017 года — всего 34 месяца.
        Это космические наблюдения, поэтому условия наземной съёмки, например высота телескопа над уровнем моря,
        к нашим данным напрямую не относятся.
    </p>
</div>

<div class="gaia-info" style="--accent:#C9B6FF;">
    <div class="gaia-note">ФАКТОРЫ, ВЛИЯЮЩИЕ НА НАБЛЮДЕНИЯ</div>
    <p>
        На наблюдаемый свет влияют температура и химический состав атмосферы звезды, расстояние до неё
        и межзвёздная пыль. Температура влияет на распределение энергии по спектру, химический состав —
        на спектральные линии, а пыль ослабляет свет и изменяет наблюдаемый цвет. Точность измерений также
        зависит от яркости объекта и особенностей его наблюдений.
    </p>
</div>

<div class="gaia-info" style="--accent:#7EE0C3;">
    <div class="gaia-note">ГРАНИЦЫ ИССЛЕДОВАНИЯ</div>
    <p>
        В нашей таблице нет отдельных данных о температуре, составе и количестве пыли для каждого объекта,
        поэтому эти факторы помогают объяснить возможные причины различий, но их влияние мы здесь отдельно не измеряем.
    </p>
</div>

<div class="gaia-info" style="--accent:#F7768E;">
    <div class="gaia-note">КАЧЕСТВО ДАННЫХ</div>
    <p>
        В коде проверяются заполненность трёх основных столбцов и отсутствие повторных
        <code style="color:#C9B6FF;">source_id</code>. Дополнительные ограничения по
        <code style="color:#B9D5FF;">ruwe</code> и ошибке параллакса не вводятся.
        Следовательно, в базе остаются измерения с разной точностью – это нужно учитывать при обсуждении результатов.
    </p>
</div>
""")

# %% [9]
# @title Загрузка общей таблицы Gaia DR3

import io
import hashlib
from pathlib import Path
import pandas as pd

file_path = Path("gaia_team.csv")

if file_path.exists():
    filename = file_path.name
    raw_bytes = file_path.read_bytes()
else:
    from google.colab import files
    uploaded = files.upload()
    filename = next(iter(uploaded))
    raw_bytes = uploaded[filename]

raw = pd.read_csv(
    io.BytesIO(raw_bytes),
    dtype={"source_id": "string"}
)

PROVENANCE = {
    "catalogue": "Gaia DR3",
    "table": "gaiadr3.gaia_source",
    "query": """SELECT source_id, random_index, ra, dec, phot_g_mean_mag, bp_rp,
       parallax, parallax_error, ruwe, phot_g_mean_flux_over_error
FROM gaiadr3.gaia_source
WHERE random_index < 20000
  AND phot_g_mean_mag IS NOT NULL
  AND bp_rp IS NOT NULL
  AND parallax IS NOT NULL
ORDER BY random_index""",
    "retrieved_utc": "2026-10-18",
    "raw_sha256": hashlib.sha256(raw_bytes).hexdigest()
}

cosmic_card("""
<div style="color:#AFC4E8;font-size:12px;letter-spacing:2px;font-weight:bold;">
    DATA IMPORT · GAIA DR3
</div>

<h2 style="color:#C9B6FF;font-size:26px;margin:8px 0 12px;">
    Общая таблица загружена
</h2>

<p style="color:#DCE6FF;margin:0;">
    Файл с данными успешно прочитан. Ниже представлены основные сведения о загруженной таблице.
</p>
""")

file_info = pd.DataFrame({
    "Параметр": ["Файл", "Размер таблицы", "Столбцы"],
    "Значение": [
        filename,
        f"{raw.shape[0]:,} строк × {raw.shape[1]} столбцов",
        ", ".join(raw.columns)
    ]
})

styled_html_table(file_info, title="Сведения о загруженном файле", digits=0)

# %% [10]
# @title Подготовка данных и контроль качества

main_columns = ['phot_g_mean_mag', 'bp_rp', 'parallax']

for column in raw.columns:
    if column != 'source_id':
        raw[column] = pd.to_numeric(raw[column], errors='coerce')

valid = np.isfinite(raw[main_columns].to_numpy(dtype=float)).all(axis=1)
prepared = raw.loc[valid].copy()

duplicates = int(prepared['source_id'].duplicated().sum())

if duplicates:
    raise ValueError(f'Повторные source_id: {duplicates}')

if len(prepared) < 10000:
    raise ValueError('В выгрузке меньше 10 000 полных строк.')

df = prepared.sort_values(
    'random_index', kind='stable'
).head(10000).reset_index(drop=True)

N = len(df)

df.to_csv(OUT/'gaia_team.csv', index=False)

PROVENANCE['team_rows'] = N
PROVENANCE['selection'] = 'First 10000 complete rows ordered by archive random_index'

(OUT/'provenance.json').write_text(
    json.dumps(PROVENANCE, ensure_ascii=False, indent=2),
    encoding='utf-8'
)

(OUT/'gaia_query.sql').write_text(
    PROVENANCE['query'],
    encoding='utf-8'
)

quality = pd.DataFrame({
    'Проверка': [
        'Строк в загруженном CSV',
        'Полных строк в загруженном CSV',
        'Строк в рабочей базе',
        'Повторных source_id',
        'Отрицательных параллаксов'
    ],
    'Результат': [
        len(raw),
        len(prepared),
        N,
        duplicates,
        int((df.parallax < 0).sum())
    ]
})

quality.to_csv(OUT/'data_quality.csv', index=False)

cosmic_card("""
<h2 style="color:#C9B6FF;font-size:25px;margin:0 0 12px;">
    Контроль качества данных
</h2>
<p style="color:#DCE6FF;margin:0;">
    Проверим полноту записей, отсутствие повторов и количество отрицательных измерений параллакса.
</p>
""")

styled_html_table(
    quality,
    title="Результаты проверки",
    digits=0
)

# %% [11]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
2. Что в работе считается случайной величиной
</h2></div>
"""))

# %% [12]
# @title Случайный опыт и случайные величины

cosmic_card("""
<style>
.random-intro {
    color:#DCE6FF;
    font-size:17px;
    line-height:1.9;
    margin-bottom:20px;
}

.random-formula {
    margin:22px 0 28px;
    padding:24px;
    border-radius:15px;
    text-align:center;
    background:linear-gradient(120deg,rgba(139,123,216,.22),rgba(65,150,230,.12));
    border:1px solid rgba(180,160,255,.35);
    box-shadow:inset 0 0 25px rgba(139,123,216,.07);
}

.random-formula .formula {
    color:#E0D7FF;
    font-size:25px;
    font-weight:600;
}

.random-formula .caption {
    color:#AFC4E8;
    font-size:13px;
    letter-spacing:2px;
    margin-bottom:12px;
}

.random-values {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
    gap:12px;
    margin:18px 0 28px;
}

.random-value {
    padding:22px 12px;
    text-align:center;
    border-radius:13px;
    background:rgba(65,150,230,.10);
    border:1px solid rgba(120,190,255,.25);
}

.random-value .symbol {
    color:#C9B6FF;
    font-size:25px;
    font-weight:bold;
    margin-bottom:10px;
}

.random-value .description {
    color:#C8D6F0;
    font-size:14px;
    line-height:1.6;
}

.random-block {
    display:flex;
    gap:15px;
    align-items:flex-start;
    margin:15px 0;
    padding:20px 22px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.18);
}

.random-block .marker {
    flex-shrink:0;
    width:9px;
    min-height:100%;
    border-radius:8px;
    background:var(--accent);
}

.random-block p {
    margin:0;
    color:#F4F5FF;
    font-size:17px;
    line-height:1.9;
}

.random-divider {
    height:1px;
    margin:28px 0;
    background:linear-gradient(90deg,transparent,#8B7BCB,transparent);
}
</style>

<p class="random-intro" style="margin:0 0 16px;">
  В таблице уже записаны результаты измерений. Случайный опыт состоит в том,
  что мы равновероятно выбираем один объект из рабочей базы.
</p>

<p class="random-intro" style="margin:0;">
  Обозначим номер выбранной строки через <i>I</i>. Каждая из 10 000 строк имеет одинаковую вероятность выбора:
</p>

<div class="random-formula">
    <div class="caption">ВЕРОЯТНОСТЬ ВЫБОРА ОБЪЕКТА</div>
    <div class="formula">
        P(I = i) = <span style="white-space:nowrap;">1 / N</span>,
        &nbsp; N = 10 000.
    </div>
</div>

<p class="random-intro" style="margin:0;">
  После выбора смотрим характеристики попавшегося объекта:
</p>

<div class="random-values">
    <div class="random-value">
        <div class="symbol"><i>G</i> = <i>G</i><sub>I</sub></div>
        <div class="description">Звёздная величина</div>
    </div>
    <div class="random-value">
        <div class="symbol"><i>C</i></div>
        <div class="description">(BP − RP)<sub>I</sub></div>
    </div>
    <div class="random-value">
        <div class="symbol">Π</div>
        <div class="description">parallax<sub>I</sub></div>
    </div>
</div>

<div class="random-divider"></div>

<div class="random-block" style="--accent:#BB9AF7;">
    <div class="marker"></div>
    <p>
      До выбора неизвестно, какая строка попадётся и какие значения мы получим.
      После выбора все три значения можно прочитать в таблице. Именно случайный выбор
      объекта задаёт здесь случайные величины.
    </p>
</div>

<div class="random-block" style="--accent:#7AA2F7;">
    <div class="marker"></div>
    <p>
      Ошибки измерений — отдельная причина неопределённости. Даже если бы все характеристики
      были измерены совершенно точно, случайный выбор объекта из неоднородной базы всё равно
      давал бы случайные значения.
    </p>
</div>

<div class="random-block" style="--accent:#7EE0C3;">
    <div class="marker"></div>
    <p>
      Наше объяснение относится к выбору из этих 10 000 записей. Оно не означает,
      что распределения в нашей базе совпадают с распределениями всех звёзд.
    </p>
</div>
""")

# %% [13]
# @title Общие определения статистик и моделей

cosmic_card("""
<style>
.stats-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 20px;
}

.stats-formulas {
    margin:20px 0 26px;
    padding:24px 18px;
    border-radius:13px;
    background:rgba(120,100,220,.13);
    border:1px solid rgba(180,160,255,.28);
    color:#E4E8FF;
    font-size:19px;
    line-height:1.8;
    overflow-x:auto;
}

.stats-note {
    margin-top:20px;
    padding:18px 22px;
    border-radius:12px;
    background:rgba(65,150,230,.11);
    border:1px solid rgba(120,190,255,.25);
    color:#DCE6FF;
    font-size:16px;
    line-height:1.8;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 20px;
">
  Общие определения статистик и моделей
</h2>

<p class="stats-text">
  Для каждой величины считаются среднее, медиана, дисперсия, стандартное отклонение,
  квартили, асимметрия и эксцесс. Дисперсия с делителем (N) описывает разброс всей конечной базы,
  а дисперсия с делителем (N-1) — стандартная исправленная оценка по выборке:
</p>

<div class="stats-formulas">

  <div style="margin:16px 0; text-align:center; white-space:nowrap;">
    <span style="font-style:italic;">μ</span><sub>N</sub> =
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;line-height:1.15;margin:0 5px;">
      <span style="border-bottom:1px solid #E4E8FF;padding:0 8px;">1</span>
      <span>N</span>
    </span>
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;line-height:1.05;margin:0 4px;">
      <span style="font-size:12px;">N</span>
      <span style="font-size:25px;">∑</span>
      <span style="font-size:12px;">i=1</span>
    </span>
    x<sub>i</sub>
  </div>

  <div style="margin:22px 0; text-align:center; white-space:nowrap;">
    <span style="font-style:italic;">v</span><sub>N</sub> =
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;line-height:1.15;margin:0 5px;">
      <span style="border-bottom:1px solid #E4E8FF;padding:0 8px;">1</span>
      <span>N</span>
    </span>
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;line-height:1.05;margin:0 4px;">
      <span style="font-size:12px;">N</span>
      <span style="font-size:25px;">∑</span>
      <span style="font-size:12px;">i=1</span>
    </span>
    (x<sub>i</sub> − μ<sub>N</sub>)<sup>2</sup>
  </div>

  <div style="margin:22px 0; text-align:center; white-space:nowrap;">
    s<sup>2</sup> =
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;text-align:center;line-height:1.15;margin:0 5px;">
      <span style="border-bottom:1px solid #E4E8FF;padding:0 8px;">1</span>
      <span>n − 1</span>
    </span>
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;line-height:1.05;margin:0 4px;">
      <span style="font-size:12px;">n</span>
      <span style="font-size:25px;">∑</span>
      <span style="font-size:12px;">i=1</span>
    </span>
    (x<sub>i</sub> − x̄)<sup>2</sup>
  </div>

</div>

<p class="stats-text">
  Для формы распределения сравниваются нормальная модель, распределение Лапласа
  и асимметричная нормальная модель. Кроме того, для конечной базы всегда существует точное
  <strong style="color:#C9B6FF;">эмпирическое распределение</strong>:
  каждая строка имеет вероятность (1/N). Параметрические модели являются приближениями,
  а эмпирическое распределение точно описывает случайный выбор из этой базы.
</p>

<div class="stats-note">
  <i>Параметрические распределения и статистические функции рассчитываются с использованием SciPy [6].</i>
</div>
""")

# %% [14]
# @title Общие функции анализа и построения графиков

FEATURES = [
    {
        'key': 'G',
        'column': 'phot_g_mean_mag',
        'label': 'G',
        'long': 'Видимая звёздная величина G',
        'unit': 'mag',
        'color': COLORS['G']
    },
    {
        'key': 'BP−RP',
        'column': 'bp_rp',
        'label': 'BP − RP',
        'long': 'Показатель цвета BP − RP',
        'unit': 'mag',
        'color': COLORS['BP−RP']
    },
    {
        'key': 'Параллакс',
        'column': 'parallax',
        'label': 'Параллакс',
        'long': 'Измеренный параллакс',
        'unit': 'mas',
        'color': COLORS['Параллакс']
    },
]


def holm_adjust(pvalues):
    """Поправка Холма для группы p-значений."""

    pvalues = np.asarray(pvalues, dtype=float)

    if pvalues.ndim != 1:
        raise ValueError('pvalues должен быть одномерным массивом.')

    if not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError('p-значения должны находиться в диапазоне [0, 1].')

    if len(pvalues) == 0:
        return np.array([], dtype=float)

    order = np.argsort(pvalues)

    adjusted_sorted = np.minimum(
        1.0,
        np.maximum.accumulate(
            pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))
        )
    )

    adjusted = np.empty_like(pvalues)
    adjusted[order] = adjusted_sorted

    return adjusted


def describe_feature(values, unit, label):
    """Рассчитывает описательные статистики и отмечает выбросы."""

    a = np.asarray(values, dtype=float)

    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError(f'Для {label} нужны непустые конечные значения.')

    q1, med, q3 = np.quantile(a, [.25, .5, .75])
    iqr = q3 - q1

    flags = (a < q1 - 1.5 * iqr) | (a > q3 + 1.5 * iqr)

    table = pd.DataFrame({
        'Показатель': [
            'Число объектов',
            'Среднее',
            'Медиана',
            'Дисперсия базы (N)',
            'Исправленная дисперсия',
            'Стандартное отклонение',
            'Минимум',
            'Q1',
            'Q3',
            'Максимум',
            'IQR',
            'Асимметрия',
            'Эксцесс',
            'Отмечено правилом 1.5 IQR'
        ],
        'Значение': [
            len(a),
            a.mean(),
            med,
            a.var(ddof=0),
            a.var(ddof=1),
            a.std(ddof=1),
            a.min(),
            q1,
            q3,
            a.max(),
            iqr,
            stats.skew(a, bias=False),
            stats.kurtosis(a, bias=False),
            int(flags.sum())
        ],
        'Единицы': [
            'объектов',
            unit,
            unit,
            unit + '²',
            unit + '²',
            unit,
            unit,
            unit,
            unit,
            unit,
            unit,
            'безразмерная',
            'безразмерный',
            'объектов'
        ]
    })

    return table, flags


def fit_models_train_test(values, key):
    """Обучает распределения на train и проверяет их на test."""

    a = np.asarray(values, dtype=float)

    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError(f'Для {key} нужны как минимум два конечных значения.')

    seed_by_key = {
        'G': 1101,
        'BP−RP': 1102,
        'Параллакс': 1103
    }

    if key not in seed_by_key:
        raise ValueError(f'Неизвестная характеристика: {key}')

    # Фиксированное разделение обеспечивает воспроизводимость.
    idx = np.random.default_rng(SEED + seed_by_key[key]).permutation(len(a))

    n_train = int(.7 * len(a))
    train = a[idx[:n_train]]
    test = a[idx[n_train:]]

    specs = {
        'Нормальная': stats.norm,
        'Лапласа': stats.laplace,
        'Асимметричная нормальная': stats.skewnorm
    }

    rows = []
    fitted = {}

    for name, dist in specs.items():
        params = tuple(float(v) for v in dist.fit(train))
        fitted[name] = (dist, params)

        loglik = float(dist.logpdf(train, *params).sum())
        pcount = len(params)

        result = stats.kstest(test, dist.cdf, args=params)

        density = np.maximum(
            dist.pdf(test, *params),
            1e-300
        )

        rows.append({
            'Модель': name,
            'Обучение, n': len(train),
            'Контроль, n': len(test),
            'Параметров': pcount,
            'AIC': 2 * pcount - 2 * loglik,
            'BIC': pcount * np.log(len(train)) - 2 * loglik,
            'KS D на контроле': result.statistic,
            'p на контроле': result.pvalue,
            'Средний log-loss': -np.log(density).mean()
        })

    table = pd.DataFrame(rows)

    table['p_Holm'] = holm_adjust(
        table['p на контроле'].to_numpy()
    )

    table['Решение при α=0,05'] = np.where(
        table['p_Holm'] < ALPHA,
        'Отвергаем H₀',
        'Нет оснований отвергать H₀'
    )

    table = table.sort_values('BIC').reset_index(drop=True)

    return table, fitted, train, test


def plot_eda(values, info, out_prefix):
    """Строит гистограмму, boxplot и эмпирическую функцию распределения."""

    a = np.asarray(values, dtype=float)

    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Для построения графиков нужны конечные значения.')

    q1, q3 = np.quantile(a, [.25, .75])
    iqr = q3 - q1
    ordered = np.sort(a)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 4),
        constrained_layout=True
    )

    # 1. Гистограмма
    axes[0].hist(
        a,
        bins='fd',
        color=info['color'],
        edgecolor='#0b1020',
        linewidth=1.2,
        density=False
    )

    axes[0].axvline(
        a.mean(),
        color=COLORS['mean'],
        linewidth=2,
        label=f'среднее = {a.mean():.3f}'
    )

    axes[0].axvline(
        np.median(a),
        color=COLORS['median'],
        linestyle='--',
        linewidth=2,
        label=f'медиана = {np.median(a):.3f}'
    )

    axes[0].set(
        xlabel=f"{info['label']}, {info['unit']}",
        ylabel='Число объектов',
        title=f"Гистограмма: {info['long']}"
    )

    axes[0].legend(frameon=False, fontsize=8)

    # 2. Диаграмма размаха
    axes[1].boxplot(
        a,
        vert=True,
        patch_artist=True,
        boxprops={
            'facecolor': '#263550',
            'edgecolor': '#A78BFA',
            'linewidth': 1.8
        },
        medianprops={
            'color': '#FF79C6',
            'linewidth': 2.5
        },
        whiskerprops={
            'color': '#7AA2F7',
            'linewidth': 1.8
        },
        capprops={
            'color': '#7AA2F7',
            'linewidth': 1.8
        },
        flierprops={
            'marker': 'o',
            'markerfacecolor': '#F7768E',
            'markeredgecolor': '#F8FAFC',
            'markersize': 3,
            'alpha': 0.55
        }
    )

    axes[1].set(
        ylabel=f"{info['label']}, {info['unit']}",
        title='Квартили и отмеченные значения',
        xticks=[]
    )

    # 3. Эмпирическая функция распределения
    axes[2].step(
        ordered,
        np.arange(1, len(a) + 1) / len(a),
        where='post',
        color=info['color'],
        linewidth=1.8
    )

    axes[2].set(
        xlabel=f"{info['label']}, {info['unit']}",
        ylabel='Доля объектов ≤ x',
        title='Эмпирическая функция распределения',
        ylim=(0, 1.02)
    )

    for ax in axes:
        ax.grid(alpha=.18, linestyle='--')
        ax.set_axisbelow(True)

    fig.savefig(
        OUT / (out_prefix + '_eda.png'),
        bbox_inches='tight'
    )

    plt.show()


def plot_models(values, info, fit_table, fitted, out_prefix):
    """Сравнивает эмпирическое распределение с тремя параметрическими моделями."""

    a = np.asarray(values, dtype=float)

    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('Для построения графиков нужны конечные значения.')

    data_range = a.max() - a.min()

    if data_range == 0:
        raise ValueError('Нельзя построить модели для постоянной величины.')

    grid = np.linspace(
        a.min() - .05 * data_range,
        a.max() + .05 * data_range,
        1200
    )

    ordered = np.sort(a)
    ec = np.arange(1, len(a) + 1) / len(a)

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 4.5),
        constrained_layout=True
    )

    # 1. Плотности распределений
    axes[0].hist(
        a,
        bins='fd',
        density=True,
        color=info['color'],
        edgecolor='#111827',
        linewidth=1.2,
        rwidth=0.88,
        alpha=0.95,
        label='Данные'
    )

    # 2. Эмпирическая функция распределения
    axes[1].step(
        ordered,
        ec,
        where='post',
        color=info['color'],
        linewidth=1.8,
        label='Эмпирическая CDF'
    )

    colors = {
        'Нормальная': '#F7768E',
        'Лапласа': '#E0AF68',
        'Асимметричная нормальная': '#7EE0C3'
    }

    for name, (dist, params) in fitted.items():
        axes[0].plot(
            grid,
            dist.pdf(grid, *params),
            label=name,
            color=colors[name],
            linewidth=2
        )

        axes[1].plot(
            grid,
            dist.cdf(grid, *params),
            label=name,
            color=colors[name],
            linewidth=1.8
        )

    axes[0].set(
        xlabel=f"{info['label']}, {info['unit']}",
        ylabel='Плотность',
        title='Данные и модели'
    )

    axes[1].set(
        xlabel=f"{info['label']}, {info['unit']}",
        ylabel='Доля объектов',
        title='Эмпирическая CDF и модели',
        ylim=(0, 1.03)
    )

    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=.18, linestyle='--')
        ax.set_axisbelow(True)

    fig.savefig(
        OUT / (out_prefix + '_models.png'),
        bbox_inches='tight'
    )

    plt.show()

# %% [15]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
3. Случайная величина 01 — видимая звёздная величина G
</h2></div>
"""))

# %% [16]
# @title Случайная величина

cosmic_card("""
<style>
.rv-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 18px;
}

.rv-code {
    display:inline-block;
    color:#C4B5FD;
    background:rgba(139,157,255,.12);
    padding:3px 8px;
    border-radius:5px;
    font-size:17px;
    font-family:monospace;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 22px;
">
    Случайная величина
</h2>

<p class="rv-text">
    <code class="rv-code">G</code> — видимая звёздная величина. Это логарифмическая шкала наблюдаемого потока: меньшему числу соответствует более яркий объект. Она не равна собственной светимости звезды: на результат влияют расстояние и поглощение света.
</p>

<p class="rv-text" style="margin-bottom:0;">
    В нашем опыте случайно выбирается одна строка рабочей базы. Значение G выбранного объекта и есть результат случайного опыта. До выбора мы не знаем, какой объект попадётся, хотя его измеренная звёздная величина уже записана в таблице. Поэтому случайность здесь связана со способом выбора объекта.
</p>
""")

# %% [17]
# @title Видимая звёздная величина G

info = FEATURES[0]
a = df[info['column']].to_numpy(float)

summary_G, flags_G = describe_feature(
    a,
    info['unit'],
    info['label']
)

summary_G.to_csv(
    OUT / 'G_summary.csv',
    index=False
)

# Описательная статистика
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 12px;
">
    Описательная статистика
</h2>

<p style="
    color:#DCE6FF;
    font-size:17px;
    line-height:1.8;
    margin:0;
">
    Видимая звёздная величина G: основные числовые характеристики,
    описывающие положение и разброс значений в рабочей базе.
</p>
""")

styled_html_table(
    summary_G,
    title='Характеристики распределения G',
    digits=3
)

# Графики
plot_eda(a, info, 'G')

# Оформленный вывод
mean_G = a.mean()
median_G = np.median(a)
outliers_G = int(flags_G.sum())

cosmic_card(f"""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по звёздной величине G
</h2>

<div style="
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:14px;
    margin-bottom:22px;
">
    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(122,162,247,.13);
        border:1px solid rgba(122,162,247,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Среднее
        </div>
        <div style="color:#E0AF68;font-size:24px;font-weight:bold;">
            {mean_G:.4f}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">mag</div>
    </div>

    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(187,154,247,.13);
        border:1px solid rgba(187,154,247,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Медиана
        </div>
        <div style="color:#BB9AF7;font-size:24px;font-weight:bold;">
            {median_G:.4f}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">mag</div>
    </div>

    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(247,118,142,.11);
        border:1px solid rgba(247,118,142,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Отмечено по правилу 1.5 IQR
        </div>
        <div style="color:#F7768E;font-size:24px;font-weight:bold;">
            {outliers_G:,}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">объектов</div>
    </div>
</div>

<div style="
    padding:20px 22px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.18);
    color:#F4F5FF;
    font-size:17px;
    line-height:1.9;
">
    Медиана выше среднего. Это согласуется с наличием хвоста
    распределения в сторону меньших значений G — более ярких объектов.

    <p style="margin:14px 0 0;">
        <b style="color:#C9B6FF;">Выбросы по правилу 1.5 IQR:</b>
        {outliers_G:,}. Эти наблюдения не удаляются: среди них могут быть реальные яркие объекты.
    </p>
</div>
""")

# %% [18]
# @title Нормализация G

cosmic_card("""
<style>
.norm-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 20px;
}

.norm-formulas {
    display:flex;
    flex-wrap:wrap;
    justify-content:space-around;
    align-items:center;
    gap:24px;
    margin:22px 0 28px;
    padding:26px 16px;
    border-radius:14px;
    background:rgba(120,100,220,.13);
    border:1px solid rgba(180,160,255,.28);
    color:#E4E8FF;
    font-size:22px;
    line-height:1.8;
    overflow-x:auto;
}

.norm-formulas > div {
    white-space:nowrap;
}

.norm-note {
    padding:20px 22px;
    border-radius:13px;
    background:rgba(65,150,230,.11);
    border:1px solid rgba(120,190,255,.25);
    color:#DCE6FF;
    font-size:17px;
    line-height:1.9;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 20px;
">
    Нормализация G
</h2>

<p class="norm-text">
    Для сравнения признаков с разными единицами используем два преобразования:
</p>

<div class="norm-formulas">
    <div>
        u =
        <span style="
            display:inline-flex;
            flex-direction:column;
            vertical-align:middle;
            text-align:center;
            line-height:1.25;
        ">
            <span style="border-bottom:1px solid #E4E8FF;padding:0 8px;">
                G − G<sub>min</sub>
            </span>
            <span>G<sub>max</sub> − G<sub>min</sub></span>
        </span>
    </div>

    <div>
        z =
        <span style="
            display:inline-flex;
            flex-direction:column;
            vertical-align:middle;
            text-align:center;
            line-height:1.25;
        ">
            <span style="border-bottom:1px solid #E4E8FF;padding:0 8px;">
                G − Ḡ
            </span>
            <span>s</span>
        </span>
    </div>
</div>

<div class="norm-note">
    После min–max все значения лежат от 0 до 1. После стандартизации среднее равно 0,
    а исправленное стандартное отклонение равно 1. Эти преобразования меняют шкалу,
    но не делают распределение нормальным автоматически.
</div>
""")

# %% [19]
# @title Масштабирование и проверка нормализации

def scale_feature(values, info, prefix):
    """Выполняет min–max-нормализацию и стандартизацию признака."""

    a = np.asarray(values, dtype=float)

    if len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Нужны как минимум два конечных значения.')

    value_range = a.max() - a.min()
    std = a.std(ddof=1)

    if value_range == 0:
        raise ValueError('Min–max нормализация невозможна: все значения одинаковы.')

    if std == 0:
        raise ValueError('Стандартизация невозможна: стандартное отклонение равно нулю.')

    # 1. Масштабирование
    u = (a - a.min()) / value_range

    # 2. Стандартизация
    z = (a - a.mean()) / std

    result = pd.DataFrame({
        'Исходное': a,
        'Min–max': u,
        'Z': z
    })

    result.to_csv(
        OUT / (prefix + '_scaled.csv'),
        index=False
    )

    # 3. Проверка результатов
    check = pd.DataFrame({
        'Показатель': [
            'Min–max min',
            'Min–max max',
            'Z среднее',
            'Z s (N−1)'
        ],
        'Значение': [
            u.min(),
            u.max(),
            z.mean(),
            z.std(ddof=1)
        ]
    })

    styled_html_table(
        check,
        title='Проверка масштабирования',
        digits=8
    )

    # 4. Контрольные условия
    assert np.isclose(u.min(), 0)
    assert np.isclose(u.max(), 1)
    assert np.isclose(z.mean(), 0, atol=1e-12)
    assert np.isclose(z.std(ddof=1), 1)

    # 5. Сравнение распределений до и после преобразований
    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 3.7),
        constrained_layout=True
    )

    plot_data = [
        (a, f"{info['label']}, {info['unit']}"),
        (u, 'Min–max'),
        (z, 'Z')
    ]

    for axis, (values_scaled, label) in zip(axes, plot_data):
        axis.hist(
            values_scaled,
            bins=50,
            color=info['color'],
            edgecolor='#111827',
            linewidth=1.2,
            rwidth=0.88,
            density=True,
            alpha=0.95
        )

        axis.set(
            xlabel=label,
            ylabel='Плотность',
            title=label
        )

        axis.grid(
            axis='y',
            alpha=0.18,
            linestyle='--'
        )

        axis.set_axisbelow(True)

    fig.savefig(
        OUT / (prefix + '_scaling.png'),
        bbox_inches='tight'
    )

    plt.show()

    return result


scaled_G = scale_feature(a, info, 'G')

# %% [20]
# @title Математическая модель распределения G

cosmic_card("""
<style>
.model-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 20px;
}

.model-indicators {
    display:flex;
    flex-wrap:wrap;
    justify-content:space-around;
    align-items:center;
    gap:14px;
    margin:24px 0;
    padding:22px 14px;
    border-radius:14px;
    background:rgba(120,100,220,.13);
    border:1px solid rgba(180,160,255,.28);
}

.model-indicator {
    min-width:85px;
    padding:12px;
    text-align:center;
    color:#D8CCFF;
    font-size:23px;
    font-weight:bold;
}

.model-note {
    margin:22px 0;
    padding:22px;
    border-radius:13px;
    background:rgba(65,150,230,.11);
    border:1px solid rgba(120,190,255,.25);
    color:#DCE6FF;
    font-size:17px;
    line-height:1.9;
}

.model-rule {
    margin:15px 0 0;
    padding:14px 18px;
    border-radius:10px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.15);
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 22px;
">
    Математическая модель распределения G
</h2>

<p class="model-text">
    Мы сравниваем три модели, чтобы проверить, насколько простое распределение может описать наши данные.
    Нормальная модель симметрична, распределение Лапласа имеет более выраженный центральный пик и тяжёлые хвосты,
    а асимметричная нормальная модель допускает скошенность.
</p>

<p class="model-text">
    Код случайно разделяет 10 000 значений на две части: по 7000 оценивает параметры моделей,
    а по оставшимся 3000 проверяет согласие с данными. Так мы проверяем модель на значениях,
    которые не использовались для её подгонки.
</p>

<p class="model-text">
    AIC и BIC рассчитываются по обучающей части. Они позволяют сравнить качество подгонки
    с учётом числа параметров: меньшие значения предпочтительнее среди рассматриваемых моделей.
    Однако даже лучшая из трёх моделей может плохо описывать данные.
</p>

<div class="model-indicators">
    <div class="model-indicator">AIC</div>
    <div class="model-indicator">BIC</div>
    <div class="model-indicator">D</div>
    <div class="model-indicator">p<sub>Holm</sub></div>
</div>

<p class="model-text">
    Критерий Колмогорова — Смирнова сравнивает эмпирическую функцию распределения контрольных данных
    с функцией распределения подогнанной модели. Чем больше статистика D, тем сильнее максимальное
    расхождение между ними. Для трёх проверок применяется поправка Холма.
</p>

<div class="model-note">
    <h3 style="
        color:#C9B6FF;
        font-size:20px;
        margin:0 0 14px;
    ">
        Правило принятия решения
    </h3>

    <div class="model-rule">
        Если <strong>p<sub>Holm</sub> &lt; 0,05</strong>, отвергаем гипотезу о соответствии проверяемой модели.
    </div>

    <div class="model-rule">
        Если <strong>p<sub>Holm</sub> ≥ 0,05</strong>, оснований отвергать её недостаточно.
        Это не доказывает правильность модели.
    </div>
</div>

<p class="model-text">
    Уровень значимости 0,05 задаёт допустимый риск ошибочного отвержения верной гипотезы
    при выполнении условий проверки.
</p>

<p class="model-text" style="margin-bottom:0;">
    Разделение данных устраняет прямую подгонку параметров по контрольным значениям.
    При этом результаты проверки рассматриваем как приближённые: реальные наблюдения и конечная
    рабочая база могут отличаться от идеальных условий статистического критерия.
</p>
""")

# %% [21]
# @title

IMAGE_URL = "https://raw.githubusercontent.com/sofia28s/statistics/refs/heads/main/kakovaVero.jpg"

display(HTML(f"""
<div style="
    width:100%;
    box-sizing:border-box;
    margin:25px 0;
    padding:35px;
    border-radius:22px;
    background-image:
        linear-gradient(rgba(5,10,30,.25),rgba(5,10,30,.25)),
        url('{BACKGROUND_URL}');
    background-size:cover;
    background-position:center;
    text-align:center;
">
    <img src="{IMAGE_URL}"
         style="
             width:250px;
             max-width:100%;
             height:auto;
             border-radius:12px;
         ">
</div>
"""))

# %% [22]
# @title Параметры моделей распределения

cosmic_card("""
<style>
.params-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 20px;
}

.params-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:16px;
    margin:22px 0 28px;
}

.params-item {
    padding:22px;
    border-radius:13px;
    background:rgba(120,100,220,.13);
    border:1px solid rgba(180,160,255,.28);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.7;
}

.params-item:nth-child(2) {
    background:rgba(65,150,230,.11);
    border-color:rgba(120,190,255,.25);
}

.params-symbol {
    font-size:24px;
    color:#C9B6FF;
    font-weight:bold;
    margin-bottom:10px;
}

.params-item:nth-child(2) .params-symbol {
    color:#9ED4FF;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 20px;
">
    Параметры моделей распределения
</h2>

<p class="params-text">
    Параметры моделей приведены отдельно. <i>loc</i> задаёт положение распределения,
    <i>scale</i> — его масштаб, а <i>shape</i> — форму асимметричной нормальной модели.
</p>

<div class="params-grid">
    <div class="params-item">
        <div class="params-symbol">loc</div>
        Положение распределения
    </div>

    <div class="params-item">
        <div class="params-symbol">scale</div>
        Масштаб распределения
    </div>

    <div class="params-item">
        <div class="params-symbol">shape</div>
        Форма асимметричной нормальной модели
    </div>
</div>

<p class="params-text">
    У нормального распределения <i>loc</i> совпадает со средним, а <i>scale</i> —
    со стандартным отклонением.
</p>

<p class="params-text" style="margin-bottom:0;">
    Для распределения Лапласа <i>scale</i> не равен стандартному отклонению.
    У асимметричной нормальной модели <i>loc</i> и <i>scale</i> также нельзя
    автоматически считать средним и стандартным отклонением.
</p>
""")

# %% [23]
# @title Модели распределения звёздной величины G

models_G, fitted_G, train_G, test_G = fit_models_train_test(a, 'G')

models_G.to_csv(
    OUT / 'G_distribution_models.csv',
    index=False
)

# 1. Сравнение моделей
models_display = models_G.copy()

styled_html_table(
    models_display,
    title='Сравнение моделей распределения звёздной величины G',
    digits=5
)

# 2. Таблица параметров моделей
parameter_rows = []

for name, (_, params) in fitted_G.items():
    names = (
        ['shape', 'loc', 'scale']
        if len(params) == 3
        else ['loc', 'scale']
    )

    parameter_rows.append({
        'Модель': name,
        **dict(zip(names, params))
    })

params_G = pd.DataFrame(parameter_rows)

# Сохраняем одинаковый порядок столбцов
# и показываем отсутствующие параметры как —
preferred_order = ['Модель', 'loc', 'scale', 'shape']
params_G = params_G.reindex(columns=preferred_order)

styled_html_table(
    params_G,
    title='Параметры подобранных моделей',
    digits=5
)

# 3. Графики моделей
plot_models(
    a,
    info,
    models_G,
    fitted_G,
    'G'
)

# 4. Итоговый вывод
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по моделям распределения G
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Среди трёх проверенных моделей наименьший BIC получила
    асимметричная нормальная модель: примерно 23 285,6.
    Однако при уровне значимости 0,05 все три модели отвергаются
    по контрольным данным после поправки Холма. Поэтому преимущество
    по BIC ещё не означает, что модель достаточно хорошо описывает
    распределение G. Для воспроизведения значений нашей рабочей базы
    используем эмпирическое распределение.
</p>
""")

# %% [24]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
4. Случайная величина 02 — цвет BP − RP (участник 2)
</h2></div>
"""))

# %% [25]
# @title

cosmic_card("""
<style>
.color-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 20px;
}

.color-formula {
    margin:24px 0 28px;
    padding:26px 18px;
    border-radius:14px;
    background:rgba(120,100,220,.13);
    border:1px solid rgba(180,160,255,.28);
    text-align:center;
}

.color-formula .main {
    font-size:27px;
    font-weight:bold;
    color:#E4E8FF;
}

.color-formula .caption {
    margin-top:10px;
    color:#C5CCE8;
    font-size:16px;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 22px;
">
    Что означает цвет
</h2>

<p class="color-text">
    <strong style="color:#C9B6FF;">BP − RP</strong> — разность измерений в синем и красном
    фотометрических каналах Gaia. Большие значения обычно соответствуют более красному
    наблюдаемому свету. Цвет зависит от спектра объекта, межзвёздного поглощения и качества
    измерения. По одному цвету нельзя точно восстановить температуру или химический состав звезды.
</p>

<div class="color-formula">
    <div class="main">BP − RP</div>
    <div class="caption">
        Синий канал BP − красный канал RP
    </div>
</div>

<p class="color-text" style="margin-bottom:0;">
    Случайной величиной считается цвет объекта, выбранного случайно из общей базы.
    Наблюдаемая конечная база задаёт дискретное эмпирическое распределение:
    одинаковые значения получают суммарную вероятность.
</p>
""")

# %% [26]
# @title Как цвет связан со спектром и химическим составом

cosmic_card("""
<style>
.spectrum-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 20px;
}

.spectrum-example {
    margin-top:30px;
    padding:24px;
    border-radius:17px;
    background:linear-gradient(135deg,rgba(105,235,190,.15),rgba(40,160,150,.08));
    border:1px solid rgba(137,245,210,.48);
    box-shadow:0 0 24px rgba(91,224,183,.08);
}

.spectrum-tag {
    display:inline-block;
    padding:5px 12px;
    margin-bottom:12px;
    border-radius:20px;
    background:rgba(142,245,207,.14);
    border:1px solid rgba(142,245,207,.42);
    color:#AAF5D9;
    font-size:12px;
    letter-spacing:1px;
}

.spectrum-elements {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(90px,1fr));
    gap:10px;
    margin:22px 0;
}

.spectrum-element {
    text-align:center;
    padding:15px 10px;
    border-radius:12px;
    background:rgba(158,247,216,.10);
    border:1px solid rgba(158,247,216,.32);
}

.spectrum-element .symbol {
    font-size:25px;
    font-weight:bold;
    color:#B6F5DD;
}

.spectrum-element .name {
    font-size:14px;
    color:#D9FFF0;
    margin-top:5px;
}

.spectrum-scale {
    margin:22px 0;
    padding:20px;
    border-radius:14px;
    background:rgba(100,230,177,.08);
    border:1px solid rgba(137,245,210,.30);
}

.spectrum-gradient {
    height:18px;
    border-radius:20px;
    background:linear-gradient(90deg,
        #8c69d4 0%,
        #697de0 18%,
        #4ba6c7 35%,
        #60d5a0 50%,
        #b8dc66 65%,
        #f0c76a 80%,
        #e88c82 100%);
    box-shadow:0 0 16px rgba(112,231,177,.14);
}

.spectrum-scale-labels {
    display:flex;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:8px;
    margin-top:10px;
    font-size:13px;
    color:#C3D8D3;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:29px;
    margin:0 0 22px;
">
    Как цвет связан со спектром и химическим составом
</h2>

<p class="spectrum-text">
    BP − RP характеризует соотношение света в двух широких фотометрических каналах.
    Спектр содержит более подробную информацию: он показывает, как поток излучения
    меняется с длиной волны. Разные спектры могут давать близкий показатель цвета,
    поэтому по одному BP − RP нельзя однозначно определить химический состав.
</p>

<p class="spectrum-text">
    Атомы и ионы поглощают свет при переходах между определёнными энергетическими
    уровнями. На спектре появляются линии поглощения. Их положения и сочетания
    сопоставляют с известными переходами элементов. Содержание элементов оценивают
    с помощью моделей атмосферы, учитывающих температуру, давление, ионизацию и
    уширение линий. Поэтому глубина одной линии сама по себе ещё не показывает
    количество элемента.
</p>

<div class="spectrum-example">

    <div class="spectrum-tag">
        ПРИМЕР ИЗ АСТРОФИЗИКИ
    </div>

    <h2 style="
        color:#AAF5D9;
        margin:0 0 18px;
        font-size:25px;
        line-height:1.5;
    ">
        Пример: зелёный оттенок белого карлика
    </h2>

    <p class="spectrum-text">
        В работе Hollands и соавторов (2026) исследован цвет белого карлика
        <span style="color:#B9F7E4;font-weight:bold;">WD J1330+6435</span>.
        Этот объект был известен ранее; новая работа рассматривает его воспринимаемый
        цвет по спектру.
    </p>

    <div class="spectrum-elements">
        <div class="spectrum-element">
            <div class="symbol">Li</div>
            <div class="name">Литий</div>
        </div>

        <div class="spectrum-element">
            <div class="symbol">Na</div>
            <div class="name">Натрий</div>
        </div>

        <div class="spectrum-element">
            <div class="symbol">Ca</div>
            <div class="name">Кальций</div>
        </div>
    </div>

    <p class="spectrum-text">
        В его атмосфере обнаружены литий, натрий и кальций. Зелёный оттенок авторы
        связывают с широким поглощением натрия в оранжевой области и кальция в
        фиолетовой. В результате относительно сильнее остаётся свет около
        <span style="
            display:inline-block;
            padding:3px 10px;
            border-radius:8px;
            background:rgba(127,255,191,.16);
            border:1px solid rgba(127,255,191,.45);
            color:#A8FFD0;
            font-weight:bold;
            white-space:nowrap;
        ">500 нм</span>.
        Следовательно, необычный цвет возникает из-за изменения формы спектра
        поглощением, а не из-за того, что литий «светится зелёным».
    </p>

    <div class="spectrum-scale">
        <div style="
            color:#B8F7DF;
            font-size:13px;
            margin-bottom:13px;
            letter-spacing:.5px;
        ">
            ОБЛАСТЬ ВИДИМОГО СПЕКТРА
        </div>

        <div class="spectrum-gradient"></div>

        <div class="spectrum-scale-labels">
            <span>Фиолетовый</span>
            <span style="color:#A8FFD0;font-weight:bold;">
                ≈ 500 нм · зелёная область
            </span>
            <span>Красный</span>
        </div>
    </div>

    <p class="spectrum-text" style="margin-bottom:0;">
        Этот пример показывает, почему цвет зависит не только от температуры.
        При этом он служит иллюстрацией из литературы: мы не проверяли, входит ли
        этот белый карлик в наши 10 000 объектов, и не определяли состав объектов
        нашей базы по BP − RP. [12]
    </p>

</div>
""")

# %% [27]
# @title

info = FEATURES[1]
a = df[info['column']].to_numpy(float)

summary_color, flags_color = describe_feature(
    a, info['unit'], info['label']
)

summary_color.to_csv(
    OUT / 'BP_RP_summary.csv',
    index=False
)

# Таблица описательной статистики
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 12px;
">
    Описательная статистика
</h2>

<p style="
    color:#DCE6FF;
    font-size:17px;
    line-height:1.8;
    margin:0;
">
    Цвет BP-RP: основные числовые характеристики,
    описывающие положение и разброс значений в рабочей базе.
</p>
""")

styled_html_table(
    summary_color,
    title='Описательная статистика: цвет BP-RP',
    digits=6
)

# Графики
plot_eda(a, info, 'BP_RP')

# Итоговый вывод
mean_color = a.mean()
median_color = np.median(a)
outliers_color = int(flags_color.sum())

cosmic_card(f"""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по цвету BP-RP
</h2>

<div style="
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:14px;
    margin-bottom:22px;
">
    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(122,162,247,.13);
        border:1px solid rgba(122,162,247,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Среднее
        </div>
        <div style="color:#E0AF68;font-size:24px;font-weight:bold;">
            {mean_color:.4f}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">mag</div>
    </div>

    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(187,154,247,.13);
        border:1px solid rgba(187,154,247,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Медиана
        </div>
        <div style="color:#BB9AF7;font-size:24px;font-weight:bold;">
            {median_color:.4f}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">mag</div>
    </div>

    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(247,118,142,.11);
        border:1px solid rgba(247,118,142,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Выбросы по правилу 1.5 IQR
        </div>
        <div style="color:#F7768E;font-size:24px;font-weight:bold;">
            {outliers_color:,}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">объектов</div>
    </div>
</div>

<div style="
    padding:20px 22px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.18);
    color:#F4F5FF;
    font-size:17px;
    line-height:1.9;
">
    <b style="color:#C9B6FF;">Среднее:</b>
    {mean_color:.4f} mag
    <span style="color:#71809F;"> · </span>
    <b style="color:#C9B6FF;">Медиана:</b>
    {median_color:.4f} mag

    <p style="margin:14px 0 0;">
        <b style="color:#C9B6FF;">Выбросы по правилу 1.5 IQR:</b>
        {outliers_color:,}.
        Отмеченные значения не удаляются автоматически:
        сначала нужно определить, являются ли они ошибками
        или редкими объектами.
    </p>
</div>
""")

# Масштабирование признака для дальнейшего анализа
scaled_color = scale_feature(a, info, 'BP_RP')

# %% [28]
# @title

display(HTML(f"""
<div style="
    width:100%;
    box-sizing:border-box;
    background:
        linear-gradient(rgba(5, 10, 30, 0.25), rgba(5, 10, 30, 0.25)),
        url('{BACKGROUND_URL}');
    background-size:cover;
    background-position:center;
    padding:35px;
    margin:25px 0;
    border-radius:18px;
    text-align:center;
">
    <img
        src="https://raw.githubusercontent.com/sofia28s/statistics/main/now.jpg"
        style="
            width:300px;
            max-width:100%;
            height:auto;
            border-radius:12px;
        "
    >
</div>
"""))

# %% [29]
# @title

cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Математическая модель цвета
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Для цвета проверяются те же три семейства. Дополнительно в работе
    сохраняется эмпирическое распределение как точная модель конечной базы.
    Если все непрерывные модели отвергаются, это означает, что они плохо
    описывают именно наблюдаемую рабочую базу; это не означает, что данные
    «не имеют распределения».
</p>
""")

# %% [30]
# @title

models_color, fitted_color, train_color, test_color = fit_models_train_test(
    a, 'BP−RP'
)

models_color.to_csv(
    OUT / 'BP_RP_distribution_models.csv',
    index=False
)

# Сравнение моделей
styled_html_table(
    models_color,
    title='Сравнение моделей распределения цвета BP-RP',
    digits=6
)

# Графики сравнения моделей
plot_models(a, info, models_color, fitted_color, 'BP_RP')

# Таблица параметров моделей
parameter_rows = []

for name, (_, params) in fitted_color.items():
    names = (
        ['shape', 'loc', 'scale']
        if len(params) == 3 else ['loc', 'scale']
    )

    parameter_rows.append({
        'Модель': name,
        **dict(zip(names, params))
    })

params_color = pd.DataFrame(parameter_rows)

styled_html_table(
    params_color,
    title='Параметры моделей распределения цвета BP-RP',
    digits=5
)

# Итоговый вывод
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по распределению цвета BP-RP
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Для цвета лучший BIC получила асимметричная нормальная модель:
    примерно 12 159,0. После поправки Холма её p-value составляет 0,131,
    что больше 0,05. Поэтому оснований отвергать эту модель в проведённой
    проверке недостаточно. Нормальная модель и распределение Лапласа
    отвергаются. Асимметричную нормальную модель можно использовать как
    приближённое описание цвета, но её истинность этим не доказана.
    Для генератора по конечной базе сохраняем эмпирическое распределение.
</p>
""")

# %% [31]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
5. Случайная величина 03 — измеренный параллакс
</h2></div>
"""))

# %% [32]
# @title

cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:28px;
    margin:0 0 22px;
">
    Что означает параллакс
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 18px;
">
    Параллакс — видимое годовое угловое смещение объекта, возникающее
    из-за движения Земли вокруг Солнца. Его используют для оценки
    расстояния, но в этой работе анализируется именно
    <b style="color:#7EE0C3;">измеренный параллакс</b>, а не величина (1/π).
    При малом сигнале ошибка измерения может быть сравнима с самим
    значением, поэтому отрицательные измерения сохраняются и
    рассматриваются как часть наблюдаемой статистики.
</p>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Случайной величиной считаем параллакс объекта, равновероятно
    выбранного из рабочей базы. Пока объект не выбран, мы не знаем,
    какое значение получим. После выбора его измеренный параллакс
    известен из таблицы. Различия расстояний и ошибки измерений
    влияют на распределение значений, которое мы наблюдаем.
</p>
""")

# %% [33]
# @title

info = FEATURES[2]
a = df[info['column']].to_numpy(float)

summary_parallax, flags_parallax = describe_feature(
    a, info['unit'], info['label']
)

summary_parallax.to_csv(
    OUT / 'parallax_summary.csv',
    index=False
)

# Таблица описательной статистики
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 12px;
">
    Описательная статистика
</h2>

<p style="
    color:#DCE6FF;
    font-size:17px;
    line-height:1.8;
    margin:0;
">
    Измеренный параллакс: основные числовые характеристики,
    описывающие положение и разброс значений в рабочей базе.
</p>
""")

styled_html_table(
    summary_parallax,
    title='Описательная статистика: параллакс',
    digits=6
)

# Полный диапазон сохраняем, чтобы были видны хвосты.
fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

ax.hist(
    a,
    bins='fd',
    color=info['color']
)

ax.set(
    xlabel='Измеренный параллакс, mas',
    ylabel='Число объектов',
    title='Параллакс: полный диапазон'
)

ax.grid(axis='y', alpha=.2)

fig.savefig(
    OUT / 'parallax_hist_full.png',
    bbox_inches='tight'
)

plt.show()

# Отдельно увеличиваем центральные 98% наблюдений.
lo, hi = np.quantile(a, [.01, .99])
edges = np.linspace(lo, hi, 70)

fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)

ax.hist(
    a,
    bins=edges,
    color=info['color']
)

ax.axvline(
    a.mean(),
    color=COLORS['mean'],
    label='Среднее'
)

ax.axvline(
    np.median(a),
    color=COLORS['median'],
    linestyle='--',
    label='Медиана'
)

ax.set(
    xlim=(lo, hi),
    xlabel='Измеренный параллакс, mas',
    ylabel='Число объектов',
    title='Параллакс: увеличенный участок между 1-м и 99-м процентилями'
)

ax.legend()
ax.grid(axis='y', alpha=.2)

fig.savefig(
    OUT / 'parallax_hist_center.png',
    bbox_inches='tight'
)

plt.show()

# Итоговый вывод
mean_parallax = a.mean()
median_parallax = np.median(a)
negative_parallax = int((a < 0).sum())

cosmic_card(f"""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по параллаксу
</h2>

<div style="
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:14px;
    margin-bottom:22px;
">
    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(122,162,247,.13);
        border:1px solid rgba(122,162,247,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Средний измеренный параллакс
        </div>
        <div style="color:#E0AF68;font-size:24px;font-weight:bold;">
            {mean_parallax:.4f}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">mas</div>
    </div>

    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(187,154,247,.13);
        border:1px solid rgba(187,154,247,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Медиана
        </div>
        <div style="color:#BB9AF7;font-size:24px;font-weight:bold;">
            {median_parallax:.4f}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">mas</div>
    </div>

    <div style="
        padding:18px;
        border-radius:12px;
        background:rgba(247,118,142,.11);
        border:1px solid rgba(247,118,142,.3);
    ">
        <div style="color:#AFC4E8;font-size:14px;margin-bottom:8px;">
            Отрицательные измерения
        </div>
        <div style="color:#F7768E;font-size:24px;font-weight:bold;">
            {negative_parallax:,}
        </div>
        <div style="color:#AFC4E8;font-size:13px;">объектов</div>
    </div>
</div>

<div style="
    padding:20px 22px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.18);
    color:#F4F5FF;
    font-size:17px;
    line-height:1.9;
">
    <b style="color:#C9B6FF;">Средний измеренный параллакс:</b>
    {mean_parallax:.4f} mas

    <p style="margin:10px 0;">
        <b style="color:#C9B6FF;">Медиана:</b>
        {median_parallax:.4f} mas
    </p>

    <p style="margin:0;">
        <b style="color:#C9B6FF;">Отрицательные измерения:</b>
        {negative_parallax:,}.
        Они не удаляются, поскольку это результат измерения с ошибкой,
        а не физическое отрицательное расстояние.
    </p>
</div>
""")

# Масштабирование признака для дальнейшего анализа
scaled_parallax = scale_feature(a, info, 'parallax')

# %% [34]
# @title

cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:28px;
    margin:0 0 22px;
">
    Математическая модель параллакса
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Параметрические распределения сравниваются на контрольной части данных.
    Из-за ошибок измерения и наблюдательного отбора форма параллакса может
    отличаться от простого симметричного колокола. Поэтому для практического
    генератора снова оставляем возможность использовать точную эмпирическую
    CDF рабочей базы.
</p>
""")

# %% [35]
# @title

models_parallax, fitted_parallax, train_parallax, test_parallax = fit_models_train_test(
    a, 'Параллакс'
)

models_parallax.to_csv(
    OUT / 'parallax_distribution_models.csv',
    index=False
)

# Сравнение моделей
styled_html_table(
    models_parallax,
    title='Сравнение моделей распределения параллакса',
    digits=6
)

lo, hi = np.quantile(a, [.01, .99])
grid = np.linspace(lo, hi, 1200)
edges = np.linspace(lo, hi, 70)

# Делим на полное число наблюдений:
# значения вне показанного участка сохраняют свою вероятность.
counts, _ = np.histogram(a, bins=edges)
density = counts / (len(a) * np.diff(edges))

colors = ['#F7768E', '#E0AF68', '#7EE0C3']
styles = ['-', '--', ':']

fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

ax.bar(
    edges[:-1],
    density,
    width=np.diff(edges),
    align='edge',
    color=info['color'],
    alpha=.4,
    label='Данные'
)

for (name, (dist, params)), color, style in zip(
    fitted_parallax.items(), colors, styles
):
    ax.plot(
        grid,
        dist.pdf(grid, *params),
        color=color,
        linestyle=style,
        linewidth=2,
        label=name
    )

ax.set(
    xlim=(lo, hi),
    xlabel='Измеренный параллакс, mas',
    ylabel='Плотность',
    title='Модели параллакса: увеличенный центральный участок'
)

ax.legend()
ax.grid(alpha=.2)

fig.savefig(
    OUT / 'parallax_models_density.png',
    bbox_inches='tight'
)

plt.show()

# Функцию распределения показываем отдельно в полном диапазоне.
grid_full = np.linspace(a.min(), a.max(), 3000)
ordered = np.sort(a)

fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

ax.step(
    ordered,
    np.arange(1, len(a) + 1) / len(a),
    where='post',
    color=info['color'],
    label='Эмпирическая CDF'
)

for (name, (dist, params)), color, style in zip(
    fitted_parallax.items(), colors, styles
):
    ax.plot(
        grid_full,
        dist.cdf(grid_full, *params),
        color=color,
        linestyle=style,
        linewidth=2,
        label=name
    )

ax.set(
    xlabel='Измеренный параллакс, mas',
    ylabel='Доля объектов ≤ x',
    ylim=(0, 1.02),
    title='Функции распределения параллакса: полный диапазон'
)

ax.legend()
ax.grid(alpha=.2)

fig.savefig(
    OUT / 'parallax_models_cdf.png',
    bbox_inches='tight'
)

plt.show()

# Таблица параметров моделей
parameter_rows = []

for name, (_, params) in fitted_parallax.items():
    names = (
        ['shape', 'loc', 'scale']
        if len(params) == 3 else ['loc', 'scale']
    )

    parameter_rows.append({
        'Модель': name,
        **dict(zip(names, params))
    })

params_parallax = pd.DataFrame(parameter_rows)

styled_html_table(
    params_parallax,
    title='Параметры моделей распределения параллакса',
    digits=5
)

# Итоговый вывод
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по распределению параллакса
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Для параллакса наименьший BIC получило распределение Лапласа:
    примерно 15 838,6. Однако все три проверенные модели отвергаются
    при уровне значимости 0,05 после поправки Холма. Следовательно,
    распределение Лапласа оказалось лучшим среди рассмотренных вариантов,
    но этого недостаточно для признания его подходящей моделью.
    Для генерации измеренных параллаксов используем эмпирическое
    распределение, сохраняя отрицательные значения. Они относятся
    к результатам измерения с ошибкой и не означают отрицательного расстояния.
</p>
""")

# %% [36]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
6. Зависимость между тремя величинами
</h2></div>
"""))

# %% [37]
# @title

cosmic_card("""
<style>
.corr-text {
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 16px;
}

.corr-pairs {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:12px;
    margin:22px 0 28px;
}

.corr-pair {
    padding:17px;
    border-radius:13px;
    background:rgba(130,155,255,.10);
    border:1px solid rgba(160,180,255,.28);
}

.corr-pair-label {
    color:#B9C9FF;
    font-size:13px;
    margin-bottom:7px;
    letter-spacing:1px;
}

.corr-pair-value {
    color:#F0F2FF;
    font-size:18px;
    font-weight:bold;
}

.corr-heading {
    color:#C9B6FF;
    font-size:21px;
    margin:28px 0 12px;
}

.corr-formula {
    overflow-x:auto;
    margin:20px 0;
    padding:24px 12px;
    text-align:center;
    border-radius:13px;
    background:rgba(120,145,255,.09);
    border:1px solid rgba(160,180,255,.22);
    color:#E8EBFF;
}

.corr-note {
    margin-top:20px;
    padding:18px 20px;
    border-radius:13px;
    background:rgba(120,145,255,.08);
    border:1px solid rgba(160,180,255,.25);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
}
</style>

<h2 style="
    color:#C9B6FF;
    font-size:28px;
    margin:0 0 18px;
">
    Анализ корреляций
</h2>

<p class="corr-text">
    Проверяем три пары: G и BP − RP, G и параллакс, BP − RP и параллакс.
    Хотим понять, меняется ли одна характеристика вместе с другой.
</p>

<div class="corr-pairs">
    <div class="corr-pair">
        <div class="corr-pair-label">ПАРА 1</div>
        <div class="corr-pair-value">G ↔ BP − RP</div>
    </div>

    <div class="corr-pair">
        <div class="corr-pair-label">ПАРА 2</div>
        <div class="corr-pair-value">G ↔ параллакс</div>
    </div>

    <div class="corr-pair">
        <div class="corr-pair-label">ПАРА 3</div>
        <div class="corr-pair-value">BP − RP ↔ параллакс</div>
    </div>
</div>

<h3 class="corr-heading">Коэффициент Пирсона</h3>

<p class="corr-text">
    <b>Коэффициент Пирсона</b> измеряет линейную связь:
</p>

<div class="corr-formula">
    <div style="
        display:inline-flex;
        align-items:center;
        gap:9px;
        white-space:nowrap;
        font-size:19px;
    ">
        <i>r</i><span>=</span>

        <span style="
            display:inline-flex;
            flex-direction:column;
            align-items:center;
        ">
            <span style="padding:0 8px 10px;">
                <span style="
                    display:inline-flex;
                    flex-direction:column;
                    vertical-align:middle;
                    align-items:center;
                    margin-right:4px;
                    line-height:1;
                ">
                    <span style="font-size:10px;height:11px;">N</span>
                    <span style="font-size:25px;line-height:26px;">∑</span>
                    <span style="font-size:10px;height:11px;">i=1</span>
                </span>
                (x<sub>i</sub> − x̄)(y<sub>i</sub> − ȳ)
            </span>

            <span style="height:1px;width:100%;background:#DCE2FF;"></span>

            <span style="padding:10px 8px 0;">
                √[
                <span style="
                    display:inline-flex;
                    flex-direction:column;
                    vertical-align:middle;
                    align-items:center;
                    margin:0 2px;
                    line-height:1;
                ">
                    <span style="font-size:10px;height:11px;">N</span>
                    <span style="font-size:25px;line-height:26px;">∑</span>
                    <span style="font-size:10px;height:11px;">i=1</span>
                </span>
                (x<sub>i</sub> − x̄)<sup>2</sup>
                ·
                <span style="
                    display:inline-flex;
                    flex-direction:column;
                    vertical-align:middle;
                    align-items:center;
                    margin:0 2px;
                    line-height:1;
                ">
                    <span style="font-size:10px;height:11px;">N</span>
                    <span style="font-size:25px;line-height:26px;">∑</span>
                    <span style="font-size:10px;height:11px;">i=1</span>
                </span>
                (y<sub>i</sub> − ȳ)<sup>2</sup>]
            </span>
        </span>
    </div>
</div>

<p class="corr-text">
    Положительный знак означает, что большие значения одной величины обычно
    сопровождаются большими значениями другой. Отрицательный знак показывает
    обратное направление. Значения ближе к −1 или 1 соответствуют более
    выраженной линейной связи.
</p>

<h3 class="corr-heading">Коэффициент Спирмена</h3>

<p class="corr-text">
    <b>Коэффициент Спирмена</b> рассчитывается по рангам: вместо самих чисел
    используются их места в упорядоченном списке. Одинаковым значениям
    присваиваются средние ранги. Поэтому он оценивает монотонную связь,
    которая может быть нелинейной:
</p>

<div class="corr-formula" style="
    font-size:23px;
    color:#E8EBFF;
">
    ρ = r(R<sub>x</sub>, R<sub>y</sub>)
</div>

<h3 class="corr-heading">Проверка перестановочным тестом</h3>

<p class="corr-text">
    Для проверки используем нулевую гипотезу: две характеристики независимы.
    Код 1 999 раз перемешивает ранги второй характеристики и каждый раз заново
    считает корреляцию. При этом значения каждого столбца сохраняются,
    но исходные соответствия между объектами разрушаются.
</p>

<p class="corr-text">
    Считаем, сколько перемешиваний дали коэффициент по модулю не меньше
    наблюдаемого. Если таких случаев b, то
</p>

<div class="corr-formula" style="
    background:rgba(105,220,190,.09);
    border-color:rgba(137,245,210,.30);
    color:#BAF5E1;
    font-size:22px;
">
    p<sub>perm</sub> =
    <span style="
        display:inline-flex;
        flex-direction:column;
        vertical-align:middle;
        align-items:center;
        line-height:1.4;
    ">
        <span style="padding:0 12px 5px;">b + 1</span>
        <span style="width:100%;height:1px;background:#BAF5E1;"></span>
        <span style="padding:5px 12px 0;">1 999 + 1</span>
    </span>
</div>

<p class="corr-text">
    Прибавление единицы учитывает исходный результат и не позволяет получить
    нулевое p-value при конечном числе перемешиваний.
</p>

<h3 class="corr-heading">Поправка Холма</h3>

<p class="corr-text">
    Мы проверяем сразу три пары, поэтому применяем поправку Холма. Она
    контролирует риск хотя бы одного ошибочного отвержения верной гипотезы
    в этой группе проверок. В таблице скорректированные значения
    <code style="color:#BAF5E1;">p_Holm</code> сравниваем с уровнем значимости 0,05
</p>

<div class="corr-note">
    <p style="margin:0;">
        Нулевой коэффициент не доказывает независимость: связь может иметь
        более сложную форму. Обнаруженная корреляция также не устанавливает
        физическую причину.
    </p>
</div>
""")

# %% [38]
# @title

# Три пары характеристик
pair_specs = [
    ('G', 'BP−RP'),
    ('G', 'Параллакс'),
    ('BP−RP', 'Параллакс')
]

arrays = {
    feature['key']: df[feature['column']].to_numpy(float)
    for feature in FEATURES
}

B_PERM = 1999
pair_rows = []

for pair_index, (left, right) in enumerate(pair_specs):
    x1, x2 = arrays[left], arrays[right]

    pear = stats.pearsonr(x1, x2)
    spear = stats.spearmanr(x1, x2)

    rng = np.random.default_rng(SEED + 2000 + pair_index)
    observed = abs(float(spear.statistic))
    exceed = 0

    # Перемешиваем ранги второй характеристики
    r1 = stats.rankdata(x1)
    r2 = stats.rankdata(x2)

    for _ in range(B_PERM):
        perm = float(
            np.corrcoef(r1, rng.permutation(r2))[0, 1]
        )
        exceed += abs(perm) >= observed

    pair_rows.append({
        'Пара': f'{left} — {right}',
        'Pearson r': pear.statistic,
        'Pearson p': pear.pvalue,
        'Spearman ρ': spear.statistic,
        'Spearman p': spear.pvalue,
        'p перестановочный': (exceed + 1) / (B_PERM + 1)
    })

dependence_table = pd.DataFrame(pair_rows)

dependence_table['p_Holm'] = holm_adjust(
    dependence_table['p перестановочный'].to_numpy()
)

dependence_table['Решение'] = np.where(
    dependence_table['p_Holm'] < ALPHA,
    'Независимость отвергается',
    'Недостаточно оснований отвергнуть независимость'
)

dependence_table.to_csv(
    OUT / 'dependence_all_pairs.csv',
    index=False
)

# Таблица результатов
styled_html_table(
    dependence_table,
    title='Корреляции и тестирование зависимости между признаками',
    digits=6
)

# Диаграммы рассеяния
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4),
    constrained_layout=True
)

for ax, (left, right) in zip(axes, pair_specs):
    ax.scatter(
        arrays[left],
        arrays[right],
        s=4,
        alpha=0.18,
        color='#527fb5',
        rasterized=True
    )

    ax.set(
        xlabel=left,
        ylabel=right,
        title=f'{left} — {right}'
    )

    ax.grid(alpha=0.15)

fig.savefig(
    OUT / 'dependence_all_pairs.png',
    bbox_inches='tight'
)

plt.show()

# Вывод
cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 18px;
">
    Вывод по анализу зависимости
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0;
">
    Во всех трёх парах получили
    <b style="color:#BAF5E1;">p_Holm = 0.0015 &lt; 0.05</b>,
    поэтому гипотезу независимости отвергаем.
</p>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:16px 0 0;
">
    При этом коэффициенты Спирмена небольшие:
    <b style="color:#C9B6FF;">0.144</b> для G и цвета,
    <b style="color:#C9B6FF;">−0.093</b> для G и параллакса,
    <b style="color:#C9B6FF;">0.129</b> для цвета и параллакса.
    Значит, обнаруженная монотонная связь
    во всех трёх случаях слабая.
</p>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:16px 0 0;
">
    Мы увидели, что маленькое p-value
    не означает сильную связь:
    при 10 000 объектах можно обнаружить
    и небольшой эффект.
    Одинаковые перестановочные p-value получились
    потому, что ни одно из 1999 перемешиваний
    не дало столь же большого по модулю коэффициента.
</p>

<div style="
    margin-top:20px;
    padding:18px 20px;
    border-radius:13px;
    background:rgba(120,145,255,.09);
    border:1px solid rgba(160,180,255,.25);
">
    <p style="
        color:#E4E8FF;
        font-size:17px;
        line-height:1.9;
        margin:0;
    ">
        Мы нашли слабые связи между характеристиками.
        Расчёт показывает, что они есть,
        но не объясняет, чем они вызваны.
    </p>
</div>
""")

# %% [39]
# @title

from IPython.display import display, HTML

DROBI_URL = "https://raw.githubusercontent.com/sofia28s/statistics/refs/heads/main/drobi.jpg"

display(HTML(f"""
<div style="
    background:
        linear-gradient(rgba(5, 10, 30, 0.25), rgba(5, 10, 30, 0.25)),
        url('{BACKGROUND_URL}');
    background-size: cover;
    background-position: center;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
">
    <img src="{DROBI_URL}"
         style="
             width:300px;
             max-width:100%;
             height:auto;
             border-radius:12px;
         ">
</div>
"""))

# %% [40]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
7. Генеральная совокупность, выборки и состоятельность оценок
</h2></div>
"""))

# %% [41]
# @title

cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:28px;
    margin:0 0 20px;
">
    Вычислительный эксперимент: точность оценок
</h2>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Для вычислительных экспериментов считаем нашу рабочую базу из <b>N = 10 000</b> строк конечной генеральной совокупностью. Её среднее и дисперсию можем рассчитать по всем строкам, а затем сравнивать с оценками по меньшим выборкам.
</p>

<h3 style="color:#C9B6FF;font-size:21px;margin:28px 0 12px;">
    Два способа отбора
</h3>

<div style="display:flex;flex-wrap:wrap;gap:14px;margin:18px 0 24px;">
    <div style="flex:1;min-width:220px;padding:18px;border-radius:13px;background:rgba(130,155,255,.10);border:1px solid rgba(160,180,255,.28);">
        <div style="font-size:17px;font-weight:bold;color:#C7D2FF;margin-bottom:8px;">С возвращением (WR)</div>
        <p style="color:#E4E8FF;font-size:17px;line-height:1.8;margin:0;">Один объект может попасть в выборку несколько раз.</p>
    </div>
    <div style="flex:1;min-width:220px;padding:18px;border-radius:13px;background:rgba(105,220,190,.09);border:1px solid rgba(137,245,210,.30);">
        <div style="font-size:17px;font-weight:bold;color:#BAF5E1;margin-bottom:8px;">Без возвращения (WOR)</div>
        <p style="color:#E4E8FF;font-size:17px;line-height:1.8;margin:0;">Внутри одной выборки повторов нет.</p>
    </div>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Обозначим среднее всей базы через <b>μ<sub>N</sub></b>, а её дисперсию с делителем <b>N</b> — через <b>v<sub>N</sub></b>:
</p>

<div style="overflow-x:auto;margin:18px 0 24px;padding:22px 14px;text-align:center;border-radius:13px;background:rgba(120,145,255,.09);border:1px solid rgba(160,180,255,.22);">
    <div style="display:flex;justify-content:center;align-items:center;flex-wrap:wrap;gap:28px;color:#E8EBFF;font-size:19px;">
        <div style="white-space:nowrap;">
            μ<sub>N</sub> =
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
                <span style="padding:0 10px 5px;">1</span>
                <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
                <span style="padding:5px 10px 0;">N</span>
            </span>
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;line-height:1;margin:0 3px;">
                <span style="font-size:10px;">N</span><span style="font-size:27px;line-height:27px;">∑</span><span style="font-size:10px;">i=1</span>
            </span>
            x<sub>i</sub>
        </div>

        <div style="white-space:nowrap;">
            v<sub>N</sub> =
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
                <span style="padding:0 10px 5px;">1</span>
                <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
                <span style="padding:5px 10px 0;">N</span>
            </span>
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;line-height:1;margin:0 3px;">
                <span style="font-size:10px;">N</span><span style="font-size:27px;line-height:27px;">∑</span><span style="font-size:10px;">i=1</span>
            </span>
            (x<sub>i</sub> − μ<sub>N</sub>)<sup>2</sup>
        </div>
    </div>
</div>

<h3 style="color:#C9B6FF;font-size:21px;margin:28px 0 12px;">
    Теоретическая дисперсия выборочного среднего
</h3>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Теоретическая дисперсия выборочного среднего равна
</p>

<div style="display:flex;flex-wrap:wrap;gap:14px;margin:18px 0 24px;">
    <div style="flex:1;min-width:230px;padding:20px 14px;text-align:center;border-radius:13px;background:rgba(130,155,255,.09);border:1px solid rgba(160,180,255,.25);">
        <div style="color:#B9C9FF;font-size:13px;margin-bottom:10px;">С ВОЗВРАЩЕНИЕМ</div>
        <div style="font-size:21px;color:#E8EBFF;white-space:nowrap;">
            Var<sub>WR</sub>(X̄<sub>n</sub>) =
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
                <span style="padding:0 10px 5px;">v<sub>N</sub></span>
                <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
                <span style="padding:5px 10px 0;">n</span>
            </span>
        </div>
    </div>

    <div style="flex:1;min-width:260px;padding:20px 14px;text-align:center;border-radius:13px;background:rgba(105,220,190,.08);border:1px solid rgba(137,245,210,.28);">
        <div style="color:#BAF5E1;font-size:13px;margin-bottom:10px;">БЕЗ ВОЗВРАЩЕНИЯ</div>
        <div style="font-size:21px;color:#E8EBFF;white-space:nowrap;">
            Var<sub>WOR</sub>(X̄<sub>n</sub>) =
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
                <span style="padding:0 10px 5px;">v<sub>N</sub></span>
                <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
                <span style="padding:5px 10px 0;">n</span>
            </span>
            ·
            <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
                <span style="padding:0 10px 5px;">N − n</span>
                <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
                <span style="padding:5px 10px 0;">N − 1</span>
            </span>
        </div>
    </div>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Индексы WR и WOR обозначают отбор с возвращением и без возвращения. Формулы согласуются с; здесь они записаны с учётом нашего определения <b>v<sub>N</sub></b> через делитель <b>N</b>.
</p>

<div style="margin:20px 0;padding:18px 20px;border-radius:13px;background:rgba(105,220,190,.08);border:1px solid rgba(137,245,210,.28);">
    <p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
        Множитель <b>(N − n)/(N − 1)</b> учитывает конечный размер базы. При отборе без возвращения разброс среднего уменьшается, а при <b>n = N</b> мы получаем всю базу и её точное среднее.
    </p>
</div>

<h3 style="color:#C9B6FF;font-size:21px;margin:28px 0 12px;">
    Повторные выборки
</h3>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Одной выбранной группы недостаточно, чтобы понять, насколько результат меняется от выборки к выборке. Поэтому для каждого размера — 20, 50, 100, 500, 2000, 5000 и 10 000 объектов — повторяем отбор 300 раз. Отдельно рассматриваем оба способа отбора.
</p>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:16px 0 0;">
    После каждого отбора вычисляем среднее и исправленную выборочную дисперсию. Затем сравниваем оценки с известными параметрами рабочей базы.
</p>

<h3 style="color:#C9B6FF;font-size:21px;margin:28px 0 12px;">
    Смещение и RMSE
</h3>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Смещение показывает, завышена или занижена оценка в среднем по повторениям. RMSE показывает размер ошибки с учётом и смещения, и разброса. Для среднего:
</p>

<div style="overflow-x:auto;margin:18px 0 22px;padding:22px 12px;text-align:center;border-radius:13px;background:rgba(120,145,255,.09);border:1px solid rgba(160,180,255,.22);">
    <div style="display:inline-flex;align-items:center;gap:8px;white-space:nowrap;color:#E8EBFF;font-size:20px;">
        RMSE(X̄) =
        <span style="display:inline-flex;align-items:stretch;">
            <span style="font-size:36px;line-height:1.2;">√</span>
            <span style="display:inline-flex;align-items:center;border-top:1px solid #DCE2FF;padding:4px 5px 0;">
                <span style="display:inline-flex;flex-direction:column;align-items:center;line-height:1;">
                    <span style="font-size:10px;">R</span><span style="font-size:26px;line-height:26px;">∑</span><span style="font-size:10px;">r=1</span>
                </span>
                <span style="margin-left:6px;">
                    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
                        <span style="padding:0 10px 5px;">(x̄<sub>r</sub> − μ<sub>N</sub>)<sup>2</sup></span>
                        <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
                        <span style="padding:5px 10px 0;">R</span>
                    </span>
                </span>
            </span>
        </span>
        <span style="font-size:16px;">, R = 300</span>
    </div>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Маленькое смещение ещё не означает, что отдельные выборки дают точные ответы: положительные и отрицательные ошибки могут взаимно компенсироваться. Поэтому рассматриваем и смещение, и RMSE.
</p>

<h3 style="color:#C9B6FF;font-size:21px;margin:28px 0 12px;">
    Оценка дисперсии
</h3>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Для оценки дисперсии учитываем способ отбора. При возвращении исправленная выборочная дисперсия оценивает <b>v<sub>N</sub></b>, а без возвращения —
</p>

<div style="margin:18px 0 22px;padding:20px;text-align:center;border-radius:13px;background:rgba(120,145,255,.09);border:1px solid rgba(160,180,255,.22);font-size:23px;color:#E8EBFF;">
    S<sup>2</sup><sub>N</sub> =
    <span style="display:inline-flex;flex-direction:column;vertical-align:middle;align-items:center;">
        <span style="padding:0 12px 5px;">N v<sub>N</sub></span>
        <span style="width:100%;border-top:1px solid #DCE2FF;"></span>
        <span style="padding:5px 12px 0;">N − 1</span>
    </span>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Именно с соответствующим значением код сравнивает результат.
</p>

<div style="margin-top:22px;padding:20px;border-radius:13px;background:rgba(130,155,255,.08);border:1px solid rgba(160,180,255,.25);">
    <p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
        Этот эксперимент показывает, как меняется точность оценок внутри нашей рабочей базы. Повторный отбор из тех же 10 000 строк не устраняет ограничения первоначального отбора из Gaia и не доказывает, что база представляет все звёзды.
    </p>
</div>
""")

# %% [42]
# @title

values = df[
    ['phot_g_mean_mag', 'bp_rp', 'parallax']
].to_numpy(float)

feature_keys = ['G', 'BP−RP', 'Параллакс']
feature_labels = ['G, mag', 'BP − RP, mag', 'Параллакс, mas']

population_rows = []

for j, key in enumerate(feature_keys):
    a = values[:, j]

    population_rows.append({
        'Характеристика': feature_labels[j],
        'N': N,
        'Среднее μ_N': a.mean(),
        'Дисперсия базы v_N': a.var(ddof=0),
        'Исправленная S²_N': a.var(ddof=1),
        's': a.std(ddof=1)
    })

population_table = pd.DataFrame(population_rows)

population_table.to_csv(
    OUT / 'population_parameters_all.csv',
    index=False
)

styled_html_table(
    population_table,
    title='Параметры конечной базы Gaia DR3',
    digits=6
)

# %% [43]
# @title

example_n = 500
example_rows = []

for mode, replace in [
    ('С возвращением', True),
    ('Без возвращения', False)
]:
    ids = np.random.default_rng(
        SEED + 300 + (1 if replace else 2)
    ).choice(N, example_n, replace=replace)

    row = {
        'Способ': mode,
        'n': example_n,
        'Уникальных объектов': len(np.unique(ids))
    }

    for j, key in enumerate(feature_keys):
        selected = values[ids, j]
        row[f'{feature_labels[j]}: среднее'] = selected.mean()
        row[f'{feature_labels[j]}: дисперсия'] = selected.var(ddof=1)

    example_rows.append(row)

examples_table = pd.DataFrame(example_rows)

examples_table.to_csv(
    OUT / 'sample_examples_all_features.csv',
    index=False
)

styled_html_table(
    examples_table,
    title='Сравнение выборок с возвращением и без возвращения',
    digits=6
)

# %% [44]
# @title

from IPython.display import display, HTML

sample_sizes = [20, 50, 100, 500, 2000, 5000, N]
REPEATS = 300
sampling_rows = []

rng = np.random.default_rng(SEED + 400)

for mode, replace in [
    ('С возвращением', True),
    ('Без возвращения', False)
]:
    for n in sample_sizes:
        mean_draws = np.empty((REPEATS, 3))
        var_draws = np.empty((REPEATS, 3))

        if not replace and n == N:
            mean_draws[:] = values.mean(axis=0)
            var_draws[:] = values.var(axis=0, ddof=1)
        else:
            for rep in range(REPEATS):
                idx = rng.choice(N, n, replace=replace)
                mean_draws[rep] = values[idx].mean(axis=0)
                var_draws[rep] = values[idx].var(axis=0, ddof=1)

        for j, key in enumerate(feature_keys):
            a = values[:, j]
            mu = a.mean()
            v = a.var(ddof=0)
            target = v if replace else a.var(ddof=1)

            finite_factor = 1.0 if replace else (N - n) / (N - 1)
            theory_se = np.sqrt(v / n * finite_factor)

            errors = mean_draws[:, j] - mu
            var_errors = var_draws[:, j] - target

            sampling_rows.append({
                'Отбор': mode,
                'n': n,
                'Характеристика': feature_labels[j],
                'R': REPEATS,
                'Смещение среднего': errors.mean(),
                'RMSE среднего': np.sqrt(np.mean(errors**2)),
                'SD средних': errors.std(ddof=1),
                'Теоретическая SE': theory_se,
                'Доля |ошибка среднего|>0.1': np.mean(
                    np.abs(errors) > 0.1
                    if key != 'Параллакс'
                    else np.abs(errors) > 0.1
                ),
                'Цель дисперсии': target,
                'Смещение дисперсии': var_errors.mean(),
                'RMSE дисперсии': np.sqrt(np.mean(var_errors**2))
            })

sampling_metrics = pd.DataFrame(sampling_rows)

sampling_metrics.to_csv(
    OUT / 'sampling_metrics_all_features.csv',
    index=False
)

styled_html_table(
    sampling_metrics[
        [
            'Отбор',
            'n',
            'Характеристика',
            'Смещение среднего',
            'RMSE среднего',
            'SD средних',
            'Теоретическая SE',
            'Смещение дисперсии',
            'RMSE дисперсии'
        ]
    ],
    title='Точность оценок при повторных выборках',
    digits=6
)

# Графики точности среднего и дисперсии
fig, axes = plt.subplots(
    3, 2,
    figsize=(13, 12),
    constrained_layout=True
)

for j, key in enumerate(feature_keys):
    for mode, color in [
        ('С возвращением', '#527fb5'),
        ('Без возвращения', '#b25371')
    ]:
        part = sampling_metrics[
            (sampling_metrics['Характеристика'] == feature_labels[j])
            & (sampling_metrics['Отбор'] == mode)
        ]

        nz = part[part['RMSE среднего'] > 0]

        axes[j, 0].plot(
            nz['n'],
            nz['RMSE среднего'],
            'o-',
            color=color,
            label=mode
        )
        axes[j, 0].plot(
            nz['n'],
            nz['Теоретическая SE'],
            '--',
            color=color,
            alpha=0.6
        )

        nzv = part[part['RMSE дисперсии'] > 0]

        axes[j, 1].plot(
            nzv['n'],
            nzv['RMSE дисперсии'],
            'o-',
            color=color,
            label=mode
        )

    axes[j, 0].set(
        xscale='log',
        yscale='log',
        xlabel='n',
        ylabel='RMSE среднего',
        title=f'Точность среднего: {feature_labels[j]}'
    )

    axes[j, 1].set(
        xscale='log',
        yscale='log',
        xlabel='n',
        ylabel='RMSE дисперсии',
        title=f'Точность дисперсии: {feature_labels[j]}'
    )

    for ax in axes[j]:
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)

fig.savefig(
    OUT / 'sampling_precision_all_features.png',
    bbox_inches='tight'
)
plt.show()

cosmic_card("""
<div style="
    padding:20px;
    border-radius:14px;
    background:rgba(130,155,255,.08);
    border:1px solid rgba(160,180,255,.25);
">
    <h3 style="
        color:#C9B6FF;
        font-size:22px;
        margin:0 0 18px;
    ">
        Вывод по повторным выборкам
    </h3>

    <p style="
        color:#E4E8FF;
        font-size:17px;
        line-height:1.9;
        margin:0;
    ">
        При увеличении размера выборки оценки в целом становятся точнее.
        Например, при отборе без возвращения переход от 20 к 500 объектам
        уменьшил RMSE среднего:<br>
        для G — с <b>0.340 до 0.067 mag</b>;<br>
        для цвета — с <b>0.132 до 0.026 mag</b>;<br>
        для параллакса — с <b>0.208 до 0.041 mas</b>.<br><br>

        В этих примерах ошибка уменьшилась примерно в пять раз.
        По одной выборке мы могли бы получить случайно удачный
        или неудачный результат, поэтому сравнивали 300 повторений.<br><br>

        При выборе всех 10 000 объектов без возвращения получаем всю базу,
        поэтому ошибка равна нулю. Эта строка есть в таблице,
        но нулевые значения не показаны на логарифмических графиках.
        При отборе с возвращением даже выборка из 10 000 строк
        может содержать повторы, поэтому точное совпадение не гарантируется.<br><br>

        Мы проверили, как размер и способ отбора влияют на точность оценок.
    </p>
</div>
""")

# %% [45]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
8. Проверка статистических гипотез
</h2></div>
"""))

# %% [46]
# @title

cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:28px;
    margin:0 0 22px 0;
">
    Проверка статистических гипотез
</h2>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    В работе выполняем три типа проверок.
</p>

<ol style="
    padding-left:26px;
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">
    <li style="margin-bottom:12px;">
        <b style="color:#D8CCFF;">В разделах 3–5</b> проверяем, согласуются ли контрольные данные с выбранной моделью распределения.
    </li>
    <li style="margin-bottom:12px;">
        <b style="color:#D8CCFF;">В разделе 6</b> проверяем независимость каждой пары характеристик с помощью перестановочного теста, использующего коэффициент Спирмена.
    </li>
    <li>
        <b style="color:#D8CCFF;">В разделе 9</b> проверяем нулевую модель случайного отбора: насколько необычны средние цвета и параллакса яркой группы по сравнению со случайными группами того же размера.
    </li>
</ol>

<div style="
    margin:22px 0;
    padding:18px 20px;
    border-radius:13px;
    background:rgba(130,155,255,.10);
    border:1px solid rgba(160,180,255,.25);
">
    <p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
        Уровень значимости заранее задаём равным
        <span style="color:#C9B6FF;font-weight:bold;">α = 0.05</span>.
        Это выбранная граница риска ошибочного отвержения верной нулевой гипотезы при выполнении условий проверки.
    </p>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    <b style="color:#D8CCFF;">P-value</b> показывает, насколько необычен наблюдаемый результат при нулевой гипотезе: вероятность получить столь же сильное или более сильное отклонение.
</p>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    В каждой рассматриваемой группе проверок применяем поправку Холма [9]. Поэтому решение принимаем по скорректированному значению
    <code style="
        color:#C9B6FF;
        background:rgba(130,155,255,.12);
        padding:3px 7px;
        border-radius:5px;
    ">p_Holm</code>:
</p>

<div style="
    margin:18px 0;
    padding:18px 22px;
    border-radius:13px;
    background:rgba(105,220,190,.07);
    border:1px solid rgba(137,245,210,.22);
">
    <ul style="
        padding-left:24px;
        margin:0;
        color:#E4E8FF;
        font-size:17px;
        line-height:2;
    ">
        <li>
            если <code style="color:#C9B6FF;">p_Holm &lt; 0.05</code>, отвергаем нулевую гипотезу;
        </li>
        <li>
            если <code style="color:#C9B6FF;">p_Holm ≥ 0.05</code>, оснований для её отвержения недостаточно.
        </li>
    </ul>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    Второй результат не доказывает, что гипотеза верна. P-value также не является вероятностью истинности гипотезы.
</p>

<div style="
    margin:22px 0;
    padding:20px;
    border-radius:13px;
    background:rgba(130,155,255,.09);
    border:1px solid rgba(160,180,255,.25);
">
    <p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0 0 14px;">
        В разделе 6 для всех трёх пар получили
        <code style="color:#C9B6FF;">p_Holm = 0.0015</code>.
        В разделе 9 для цвета и параллакса получили
        <code style="color:#C9B6FF;">p_Holm = 0.001</code>.
        Эти значения меньше 0,05, поэтому соответствующие нулевые гипотезы отвергаем.
    </p>
</div>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin-bottom:0;
">
    Статистическая значимость не показывает силу связи или размер различия, поэтому вместе с p-value рассматриваем коэффициенты корреляции и разности средних.
</p>
""")

# %% [47]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
9. Эксперимент: что меняется при выборе самых ярких объектов
</h2></div>
"""))

# %% [48]
# @title

cosmic_card("""
<div style="
    color:#B9A8FF;
    font-size:12px;
    font-weight:700;
    letter-spacing:2px;
    text-transform:uppercase;
    margin-bottom:10px;
">
    Анализ выборки
</div>

<h2 style="
    color:#C9B6FF;
    font-size:28px;
    line-height:1.35;
    margin:0 0 20px 0;
">
    Проверка смещения при отборе ярких объектов
</h2>

<p style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
    margin:0 0 26px 0;
">
    Проверяем отдельный вопрос: можно ли взять только самые яркие объекты и по ним судить о среднем цвете и параллаксе всей рабочей базы?
</p>

<div style="
    display:flex;
    flex-wrap:wrap;
    gap:14px;
    margin-bottom:26px;
">
    <div style="
        flex:1 1 200px;
        padding:22px;
        border-radius:14px;
        background:rgba(130,145,255,.10);
        border:1px solid rgba(165,180,255,.25);
    ">
        <div style="
            font-size:12px;
            color:#B9A8FF;
            font-weight:700;
            letter-spacing:1px;
            margin-bottom:8px;
        ">
            ОТБОР
        </div>
        <div style="
            font-size:34px;
            font-weight:700;
            color:#FFFFFF;
            margin-bottom:8px;
        ">
            500
        </div>
        <div style="
            font-size:16px;
            color:#E4E8FF;
            line-height:1.8;
        ">
            Выбираем объекты с наименьшими G: меньшая звёздная величина соответствует большей наблюдаемой яркости.
        </div>
    </div>

    <div style="
        flex:1 1 200px;
        padding:22px;
        border-radius:14px;
        background:rgba(105,220,190,.08);
        border:1px solid rgba(137,245,210,.25);
    ">
        <div style="
            font-size:12px;
            color:#BAF5E1;
            font-weight:700;
            letter-spacing:1px;
            margin-bottom:8px;
        ">
            СЛУЧАЙНОЕ СРАВНЕНИЕ
        </div>
        <div style="
            font-size:34px;
            font-weight:700;
            color:#FFFFFF;
            margin-bottom:8px;
        ">
            1999
        </div>
        <div style="
            font-size:16px;
            color:#E4E8FF;
            line-height:1.8;
        ">
            Для сравнения 1999 раз равновероятно выбираем из базы по 500 строк без возвращения.
        </div>
    </div>
</div>

<div style="
    border-left:3px solid #A5B4FC;
    padding:4px 0 4px 20px;
    margin-bottom:26px;
">
    <h3 style="
        font-size:21px;
        color:#C9B6FF;
        margin:0 0 10px 0;
    ">
        Как проводится сравнение
    </h3>
    <p style="
        margin:0;
        font-size:17px;
        color:#E4E8FF;
        line-height:1.9;
    ">
        Для каждой случайной группы вычисляем средние значения. Так видим, какие результаты обычно получаются при случайном отборе.
    </p>
</div>

<div style="
    border-left:3px solid #C4B5FD;
    padding:4px 0 4px 20px;
    margin-bottom:26px;
">
    <h3 style="
        font-size:21px;
        color:#C9B6FF;
        margin:0 0 10px 0;
    ">
        Оценка различий
    </h3>
    <p style="
        margin:0;
        font-size:17px;
        color:#E4E8FF;
        line-height:1.9;
    ">
        Затем сравниваем отклонение среднего яркой группы от среднего базы с отклонениями случайных групп. Для цвета и параллакса рассчитываем p-value методом Монте-Карло и применяем поправку Холма к двум проверкам.
    </p>
</div>

<div style="
    padding:22px;
    border-radius:14px;
    background:rgba(180,155,255,.10);
    border:1px solid rgba(196,181,253,.28);
">
    <div style="
        color:#C4B5FD;
        font-size:12px;
        font-weight:700;
        letter-spacing:1px;
        margin-bottom:10px;
    ">
        ВАЖНОЕ ЗАМЕЧАНИЕ
    </div>
    <p style="
        margin:0;
        font-size:17px;
        color:#F0EDFF;
        line-height:1.9;
    ">
        Различие по самой G ожидается заранее: именно по ней отбирались объекты. Поэтому основной интерес представляют изменения двух других характеристик — цвета и параллакса.
    </p>
</div>
""")

# %% [49]
# @title

BRIGHT_N = 500
SELECTION_REPEATS = 1999

bright_indices = np.argsort(values[:, 0], kind='stable')[:BRIGHT_N]
bright_values = values[bright_indices]

pop_means = values.mean(axis=0)
pop_stds = values.std(axis=0, ddof=0)

rng = np.random.default_rng(SEED + 500)
random_means = np.empty((SELECTION_REPEATS, 3))
random_medians = np.empty_like(random_means)
random_stds = np.empty_like(random_means)

for rep in range(SELECTION_REPEATS):
    idx = rng.choice(N, BRIGHT_N, replace=False)
    selected = values[idx]

    random_means[rep] = selected.mean(axis=0)
    random_medians[rep] = np.median(selected, axis=0)
    random_stds[rep] = selected.std(axis=0, ddof=1)

observed = np.abs(bright_values.mean(axis=0) - pop_means)

raw_p = []
for j in [1, 2]:
    raw_p.append(
        (1 + np.sum(
            np.abs(random_means[:, j] - pop_means[j]) >= observed[j]
        )) / (SELECTION_REPEATS + 1)
    )

holm_p = holm_adjust(raw_p)

selection_rows = []

for j, key in enumerate(feature_keys):
    low, high = np.quantile(random_means[:, j], [0.025, 0.975])

    selection_rows.append({
        'Характеристика': feature_labels[j],
        'Среднее базы': pop_means[j],
        'Среднее ярких 500': bright_values[:, j].mean(),
        'Разность': bright_values[:, j].mean() - pop_means[j],
        'Сдвиг / σ базы': (
            bright_values[:, j].mean() - pop_means[j]
        ) / pop_stds[j],
        '95% случайных средних: низ': low,
        '95% случайных средних: верх': high,
        'p_MC': np.nan if j == 0 else raw_p[j - 1],
        'p_Holm': np.nan if j == 0 else holm_p[j - 1],
        'Решение': (
            'Контроль отбора' if j == 0 else
            ('Отличается от случайной выборки'
             if holm_p[j - 1] < ALPHA
             else 'Недостаточно оснований для отличия')
        )
    })

selection_table = pd.DataFrame(selection_rows)

selection_table.to_csv(
    OUT / 'bright_selection_all_features.csv',
    index=False
)


# Отображение таблицы в HTML-стиле
from IPython.display import display, HTML
import html

styled_html_table(
    selection_table,
    title='Результаты яркостного отбора',
    digits=6
)


# Порог яркой группы
threshold = values[bright_indices, 0].max()

cosmic_card(f"""
<div style="
    padding:20px 22px;
    border-radius:14px;
    background:rgba(130,155,255,.10);
    border:1px solid rgba(160,180,255,.25);
">
    <h3 style="
        color:#C9B6FF;
        font-size:21px;
        margin:0 0 14px;
    ">
        Критерий отбора
    </h3>
    <p style="
        color:#E4E8FF;
        font-size:17px;
        line-height:1.9;
        margin:0;
    ">
        Выбрано <b style="color:#8FE3C1;">{BRIGHT_N}</b> самых ярких объектов.<br>
        Порог яркости: <b style="color:#8FE3C1;">G ≤ {threshold:.4f} mag</b>.
        Чем меньше звёздная величина G, тем ярче объект.
    </p>
</div>
""")


# Графики распределений средних случайных выборок
fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4),
    constrained_layout=True
)

for j, key in enumerate(feature_keys):
    axes[j].hist(
        random_means[:, j],
        bins=35,
        color='#d6e1ee',
        edgecolor='white'
    )

    axes[j].axvline(
        bright_values[:, j].mean(),
        color=COLORS['bright'],
        lw=2,
        label='500 самых ярких'
    )

    axes[j].axvline(
        pop_means[j],
        color=COLORS['median'],
        ls='--',
        label='Вся база'
    )

    axes[j].set(
        xlabel=feature_labels[j],
        ylabel='Число случайных выборок',
        title=feature_labels[j]
    )

    axes[j].legend(frameon=False, fontsize=8)
    axes[j].grid(alpha=.15)

fig.savefig(
    OUT / 'bright_selection_all_features.png',
    bbox_inches='tight'
)
plt.show()


# Итоговый вывод
cosmic_card("""
<h3 style="
    color:#C9B6FF;
    font-size:23px;
    margin:0 0 18px;
">
    Вывод
</h3>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin:0;">
    Отбор самых ярких объектов изменил обе проверяемые характеристики.<br><br>

    Средний цвет BP − RP уменьшился
    с <b style="color:#8FE3C1;">1.621 до 1.368 mag</b>, то есть на <b style="color:#8FE3C1;">0.253 mag</b>.
    Средний измеренный параллакс увеличился
    с <b style="color:#8FE3C1;">0.372 до 0.870 mas</b>, то есть на <b style="color:#8FE3C1;">0.498 mas</b>.<br><br>

    Для сравнения: центральные 95% средних случайных групп
    находились примерно от <b style="color:#8FE3C1;">1.572 до 1.673 mag</b> для цвета
    и от <b style="color:#8FE3C1;">0.291 до 0.450 mas</b> для параллакса.
    Средние яркой группы вышли за оба диапазона.
    Для обеих проверок <b style="color:#8FE3C1;">p_Holm = 0.001</b>.<br><br>

    Получается, что 500 самых ярких объектов отличаются
    от обычной случайной группы того же размера.
    Если использовать их средние для описания всей базы,
    получим смещение по цвету и параллаксу.<br><br>

    Когда мы оставили только самые яркие объекты, изменились и средний цвет, и средний параллакс. Поэтому по такой группе нельзя судить о средних значениях всей нашей базы.
</p>
""")

# %% [50]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
10. Зачем нужен генератор
</h2></div>
"""))

# %% [51]
# @title

cosmic_card("""
<h2 style="
    color:#C9B6FF;
    font-size:27px;
    margin:0 0 22px;
">
    Генератор псевдослучайных значений
</h2>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    Генератор должен создавать значения, похожие на наблюдаемые значения одной
    характеристики. В основной части строим генератор для G, а затем проверяем
    тот же алгоритм отдельно на цвете и параллаксе.
</p>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    Здесь важно различать две вещи:
</p>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:16px 0 24px;">

  <div style="
      padding:18px;
      border-radius:13px;
      background:rgba(100,190,255,.13);
      border:1px solid rgba(140,210,255,.38);
      color:#E4E8FF;
  ">
    <div style="color:#A8DEFF;font-weight:bold;font-size:18px;margin-bottom:8px;">
        PCG64
    </div>
    В NumPy — базовый генератор псевдослучайных чисел, который используется через
    <code style="color:#A8DEFF;">np.random.default_rng(SEED)</code> [5].
  </div>

  <div style="
      padding:18px;
      border-radius:13px;
      background:rgba(65,120,220,.16);
      border:1px solid rgba(125,165,255,.38);
      color:#E4E8FF;
  ">
    <div style="color:#B8CFFF;font-weight:bold;font-size:18px;margin-bottom:8px;">
        Наши методы
    </div>
    Алгоритмы, которые преобразуют равномерные псевдослучайные числа в значения нужной случайной величины.
  </div>

</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    PCG64 работает с внутренним состоянием, которое изменяется по определённому
    алгоритму. Упрощённо этот процесс можно записать так:
</p>

<div style="text-align:center;margin:20px 0;padding:22px;border-radius:13px;background:rgba(120,100,220,.13);border:1px solid rgba(180,160,255,.3);">
    <span style="font-family:'Times New Roman',serif;font-size:26px;color:#E4E8FF;">
        <i>s</i><sub>n+1</sub> = (<i>a s</i><sub>n</sub> + <i>c</i>) mod 2<sup>128</sup>
    </span>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    где <i>s</i><sub>n</sub> — текущее состояние генератора,
    <i>a</i> и <i>c</i> — параметры алгоритма.
</p>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    Затем PCG64 дополнительно преобразует биты состояния, чтобы получить выходное число.
    При вызове <code style="color:#9BD7FF;">rng.random()</code> мы получаем
    псевдослучайное число из равномерного распределения:
</p>

<div style="text-align:center;margin:20px 0;padding:20px;border-radius:13px;background:rgba(65,150,230,.12);border:1px solid rgba(120,190,255,.3);">
    <span style="font-family:'Times New Roman',serif;font-size:26px;color:#E4E8FF;">
        <i>U</i> ∼ <i>U</i>(0, 1)
    </span>
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;">
    Но характеристики звёзд не распределены равномерно. Поэтому дальше используем
    методы обратной CDF и суперпозиции, которые позволяют получить значения
    с распределением, соответствующим нашей базе Gaia.
</p>

<div style="
    margin:20px 0;
    padding:18px 22px;
    border-left:4px solid #a99aff;
    border-radius:8px;
    background:rgba(130,110,220,.12);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">
    <code style="color:#8FE3C1;">seed</code> используется для инициализации генератора.
    Если использовать одинаковый <code style="color:#8FE3C1;">seed</code> и повторить
    вычисления в тех же условиях, получим ту же последовательность. Это удобно для
    проверки результатов, но одновременно показывает, что псевдослучайность не равна
    полной непредсказуемости.
</div>

<p style="color:#E4E8FF;font-size:17px;line-height:1.9;margin-bottom:0;">
    Мы не выдаём сгенерированные значения за новые наблюдения звёзд. Генератор
    воспроизводит одномерные распределения рабочей базы. Если генерировать G, цвет
    и параллакс независимо, исходные зависимости между ними потеряются.
</p>
""")

# %% [52]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
11. Математическое описание алгоритмов
</h2></div>
"""))

# %% [53]
#@title Генераторы по эмпирическому распределению
# Два генератора по эмпирическому распределению

def validate_input(values, size):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or a.size==0 or not np.isfinite(a).all():
        raise ValueError('values должен быть непустым одномерным массивом конечных чисел')
    if isinstance(size,(bool,np.bool_)) or not isinstance(size,(int,np.integer)) or size<0:
        raise ValueError('size должен быть неотрицательным целым числом')
    return a

def empirical_inverse_sample(values, size, rng):
    a=validate_input(values,size); ordered=np.sort(a)
    positions=np.floor(rng.random(size)*len(ordered)).astype(np.int64)
    return ordered[positions]

def empirical_superposition_sample(values, size, rng):
    a=validate_input(values,size); pivot=np.median(a)
    left=a[a<=pivot]; right=a[a>pivot]
    if len(right)==0: return empirical_inverse_sample(left,size,rng)
    choose_right=rng.random(size)>=len(left)/len(a)
    result=np.empty(size,dtype=float)
    result[~choose_right]=empirical_inverse_sample(left,int((~choose_right).sum()),rng)
    result[choose_right]=empirical_inverse_sample(right,int(choose_right.sum()),rng)
    return result

generator_rows=[]; generated_by_method={}
for feature_index,(key,label) in enumerate(zip(feature_keys,feature_labels)):
    source=values[:,feature_index]
    for method_index,(method_name,method) in enumerate([('Обратная CDF',empirical_inverse_sample),('Суперпозиция',empirical_superposition_sample)]):
        generated=method(source,10000,np.random.default_rng(SEED+700+feature_index*10+method_index))
        generated_by_method[(key,method_name)]=generated
        generator_rows.append({'Характеристика':label,'Метод':method_name,'n':len(generated),
            'Среднее базы':source.mean(),'Среднее генератора':generated.mean(),
            'Медиана базы':np.median(source),'Медиана генератора':np.median(generated),
            's базы':source.std(ddof=1),'s генератора':generated.std(ddof=1),
            'Q1 базы':np.quantile(source,.25),'Q1 генератора':np.quantile(generated,.25),
            'Q3 базы':np.quantile(source,.75),'Q3 генератора':np.quantile(generated,.75),
            'KS D':stats.ks_2samp(source,generated).statistic,
            'Wasserstein':stats.wasserstein_distance(source,generated)})
generator_comparison=pd.DataFrame(generator_rows)
generator_comparison.to_csv(OUT/'generator_comparison_all_features.csv',index=False)

styled_html_table(
    generator_comparison.round(6),
    title='Сравнение эмпирических генераторов',
    digits=6
)

# Главный генератор G и его распределение
main_generated=generated_by_method[('G','Обратная CDF')]
pd.DataFrame({'G_generated':main_generated}).to_csv(OUT/'generated_G.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
for j,key in enumerate(feature_keys):
    source=values[:,j]; gen=generated_by_method[(key,'Обратная CDF')]
    axes[j].step(np.sort(source),np.arange(1,N+1)/N,where='post',label='База',color=COLORS['median'])
    axes[j].step(np.sort(gen),np.arange(1,len(gen)+1)/len(gen),where='post',label='Генератор',color=COLORS['G'])
    axes[j].set(xlabel=feature_labels[j],ylabel='F(x)',title=feature_labels[j],ylim=(0,1)); axes[j].legend(frameon=False,fontsize=8); axes[j].grid(alpha=.15)
fig.savefig(OUT/'generator_cdf_comparison_all_features.png',bbox_inches='tight'); plt.show()

cosmic_card(
    "<p>Вывод: обе реализации воспроизводят одномерные эмпирические распределения. Это воспроизведение конечной базы, а не прогноз свойств неизвестной звезды.</p>"
)

# %% [54]
#@title Методы генерации случайных величин

cosmic_card("""
<style>
.formula {
    box-sizing:border-box;
    width:100%;
    margin:16px 0;
    padding:20px 14px;
    border-radius:12px;
    background:rgba(65,150,230,.10);
    border:1px solid rgba(120,190,255,.22);
    color:#f2f3ff;
    text-align:center;
    font-family:"Times New Roman",serif;
    font-size:23px;
    line-height:1.5;
    overflow-x:auto;
}
.formula.purple {
    background:rgba(100,190,255,.13);
    border-color:rgba(140,210,255,.38);
}
.frac {
    display:inline-flex;
    flex-direction:column;
    vertical-align:middle;
    text-align:center;
    line-height:1.15;
    margin:0 3px;
    white-space:nowrap;
}
.frac .num {
    display:block;
    padding:0 6px 3px;
    border-bottom:1px solid currentColor;
}
.frac .den {
    display:block;
    padding:3px 6px 0;
}
.sum {
    display:inline-flex;
    flex-direction:column;
    align-items:center;
    vertical-align:middle;
    line-height:1;
    margin:0 3px;
}
.sum .upper {
    font-size:.52em;
    margin-bottom:1px;
}
.sum .symbol {
    font-size:1.65em;
    line-height:.85;
}
.sum .lower {
    font-size:.52em;
    margin-top:2px;
}
.math-note {
    padding:18px 22px;
    margin:18px 0;
    border-radius:10px;
    background:rgba(65,150,230,.10);
    border:1px solid rgba(120,190,255,.22);
}
</style>

<h2 style="color:#C9B6FF;font-size:27px;margin:0 0 20px;">
Методы генерации случайных величин
</h2>

<p>
Для генерации случайных величин мы выбрали два метода: обратную функцию распределения
и суперпозицию. Оба метода используют равномерные псевдослучайные числа, полученные
с помощью PCG64, но преобразуют их по-разному.[^4]
</p>

<h3 style="color:#C9B6FF;font-size:21px;margin-top:28px;">Метод обратной функции</h3>

<p>
Идея метода заключается в том, что равномерно распределённое число можно преобразовать
в значение случайной величины с нужным распределением.
</p>

<p>Пусть <i>F(x)</i> — функция распределения случайной величины <i>X</i>:</p>

<div class="formula purple">
<i>F</i>(<i>x</i>) = P(<i>X</i> ≤ <i>x</i>)
</div>

<p>Если <i>U</i> ∼ <i>U</i>(0, 1), то</p>

<div class="formula purple">
<i>X</i> = <i>F</i><sup>−1</sup>(<i>U</i>)
</div>

<p>
где <i>F</i><sup>−1</sup> — обратная функция распределения (в общем случае —
обобщённая обратная функция).
</p>

<div class="math-note">
    <div style="color:#8FE3C1;font-weight:bold;margin-bottom:8px;">Почему это работает:</div>
    <div style="text-align:center;font-family:'Times New Roman',serif;font-size:21px;overflow-x:auto;">
        P(<i>X</i> ≤ <i>x</i>) = P(<i>F</i><sup>−1</sup>(<i>U</i>) ≤ <i>x</i>) =
        P(<i>U</i> ≤ <i>F</i>(<i>x</i>)) = <i>F</i>(<i>x</i>)
    </div>
</div>

<p>
Получается, что новая случайная величина действительно имеет заданное распределение.
</p>

<p>
В нашей работе точная функция распределения неизвестна, поэтому используем эмпирическую:
</p>

<div class="formula">
<i>F</i><sub>N</sub>(<i>x</i>) =
<span class="frac">
    <span class="num">1</span>
    <span class="den"><i>N</i></span>
</span>
<span class="sum">
    <span class="upper"><i>N</i></span>
    <span class="symbol">∑</span>
    <span class="lower"><i>i</i> = 1</span>
</span>
<i>I</i>(<i>x</i><sub>i</sub> ≤ <i>x</i>)
</div>

<p>
где <i>N</i> — количество наблюдений, а <i>I</i> — индикатор выполнения условия.
</p>

<p>
Для генерации сортируем исходные значения, получаем случайное число <i>U</i>
и выбираем элемент с индексом
</p>

<div class="formula">
<i>k</i> = ⌊<i>UN</i>⌋
</div>

<p>
Таким образом, мы фактически случайно выбираем значения из исходной базы с возвращением.
Каждая позиция выбирается с вероятностью 1/<i>N</i>. Если одинаковое значение встречается
несколько раз, суммарная вероятность его получения увеличивается.
</p>

<div style="height:1px;background:rgba(180,195,255,.22);margin:30px 0;"></div>

<h3 style="color:#A8DEFF;font-size:21px;margin-top:0;">Метод суперпозиции</h3>

<p>
Этот метод основан на представлении распределения в виде смеси нескольких распределений:
</p>

<div class="formula">
<i>F</i>(<i>x</i>) =
<span class="sum">
    <span class="upper"><i>m</i></span>
    <span class="symbol">∑</span>
    <span class="lower"><i>j</i> = 1</span>
</span>
<i>w</i><sub>j</sub><i>F</i><sub>j</sub>(<i>x</i>),
&nbsp;&nbsp;
<span class="sum">
    <span class="upper"><i>m</i></span>
    <span class="symbol">∑</span>
    <span class="lower"><i>j</i> = 1</span>
</span>
<i>w</i><sub>j</sub> = 1
</div>

<p>
где <i>F</i><sub>j</sub>(x) — распределение отдельной компоненты, а
<i>w</i><sub>j</sub> — вероятность её выбора.
</p>

<p>Сначала случайно выбирается компонента, затем из неё генерируется значение.</p>

<p>По формуле полной вероятности:</p>

<div class="formula">
P(<i>X</i> ≤ <i>x</i>) =
<span class="sum">
    <span class="upper"><i>m</i></span>
    <span class="symbol">∑</span>
    <span class="lower"><i>j</i> = 1</span>
</span>
P(<i>J</i> = <i>j</i>)P(<i>X</i> ≤ <i>x</i> | <i>J</i> = <i>j</i>)
=
<span class="sum">
    <span class="upper"><i>m</i></span>
    <span class="symbol">∑</span>
    <span class="lower"><i>j</i> = 1</span>
</span>
<i>w</i><sub>j</sub><i>F</i><sub>j</sub>(<i>x</i>) = <i>F</i>(<i>x</i>)
</div>

<p>В нашей реализации мы делим значения по медиане на две группы:</p>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:16px 0;">
    <div style="padding:18px;text-align:center;border-radius:12px;background:rgba(100,190,255,.13);border:1px solid rgba(140,210,255,.38);font-family:'Times New Roman',serif;font-size:22px;color:#A8DEFF;">
        A = { <i>x</i><sub>i</sub> : <i>x</i><sub>i</sub> ≤ Med(<i>X</i>) }
    </div>
    <div style="padding:18px;text-align:center;border-radius:12px;background:rgba(65,120,220,.16);border:1px solid rgba(125,165,255,.38);font-family:'Times New Roman',serif;font-size:22px;color:#B8CFFF;">
        B = { <i>x</i><sub>i</sub> : <i>x</i><sub>i</sub> > Med(<i>X</i>) }
    </div>
</div>

<p>Вероятности выбора групп определяем по их размерам:</p>

<div class="formula">
<i>w</i><sub>A</sub> =
<span class="frac"><span class="num"><i>N</i><sub>A</sub></span><span class="den"><i>N</i></span></span>,
&nbsp;&nbsp;
<i>w</i><sub>B</sub> =
<span class="frac"><span class="num"><i>N</i><sub>B</sub></span><span class="den"><i>N</i></span></span>
</div>

<p>Внутри каждой группы снова используем обратную эмпирическую CDF.</p>

<p>Вероятность выбрать конкретную позицию из первой группы равна</p>

<div class="formula purple">
P(<i>I</i> = <i>i</i>) =
<span class="frac">
    <span class="num"><i>N</i><sub>A</sub></span>
    <span class="den"><i>N</i></span>
</span>
·
<span class="frac">
    <span class="num">1</span>
    <span class="den"><i>N</i><sub>A</sub></span>
</span>
=
<span class="frac">
    <span class="num">1</span>
    <span class="den"><i>N</i></span>
</span>
</div>

<p>Для второй группы получается то же самое.</p>

<div class="math-note">
    То есть оба метода в нашей реализации дают одинаковое теоретическое распределение,
    хотя сами последовательности могут отличаться.
</div>

<div style="height:1px;background:rgba(180,195,255,.22);margin:30px 0;"></div>

<h3 style="color:#C9B6FF;font-size:21px;margin-top:0;">Ограничения методов</h3>

<p>
Обратная функция достаточно проста для нашей базы, но не создаёт новых значений
за пределами наблюдений. Для сложных теоретических распределений обратную функцию
иногда трудно вычислить.
</p>

<p>
Суперпозиция удобна, если распределение состоит из нескольких компонентов.
Однако в нашем случае деление по медиане условное: оно не означает, что мы выделили
два физических типа звёзд. Поэтому суперпозиция не даёт нам принципиально нового распределения.
</p>

<p>
Кроме того, оба метода воспроизводят только распределение исходных данных и не
учитывают погрешности измерений Gaia.
</p>

<h3 style="color:#A8DEFF;font-size:21px;margin-top:30px;">Обзор дополнительных методов</h3>
""")

additional_methods = pd.DataFrame([
    {
        'Метод': 'Обратная CDF',
        'Где применяется': 'дискретные и непрерывные распределения с доступной CDF',
        'Сильная сторона': 'простое доказательство',
        'Ограничение': 'обратная функция может быть сложной'
    },
    {
        'Метод': 'Суперпозиция',
        'Где применяется': 'смеси распределений',
        'Сильная сторона': 'удобно для неоднородных данных',
        'Ограничение': 'нужны корректные веса'
    },
    {
        'Метод': 'Нейман, принятие–отклонение',
        'Где применяется': 'сложные плотности при известной огибающей',
        'Сильная сторона': 'не требуется обратная CDF',
        'Ограничение': 'бывают отказы, нужно доказать границу'
    },
    {
        'Метод': 'Ratio of uniforms',
        'Где применяется': 'непрерывные плотности',
        'Сильная сторона': 'универсальный геометрический метод',
        'Ограничение': 'нужны конечные границы области'
    }
])

styled_html_table(
    additional_methods,
    title=None,
    digits=6
)

cosmic_card("""
<p style="margin-bottom:0;">
Два основных метода реализованы для эмпирического распределения G и применяются
также к цвету BP−RP и параллаксу. Два дополнительных метода рассмотрены на примере
стандартного нормального распределения N(0,1).
</p>
""")

# %% [55]
# @title Два генератора по эмпирическому распределению

# Названия характеристик
feature_keys = ['G', 'BP-RP', 'parallax']

feature_labels = [
    'G (звёздная величина)',
    'BP − RP (цветовой индекс)',
    'Parallax (параллакс)'
]

# Проверяем исходные данные
values = np.asarray(values, dtype=float)

if values.ndim != 2 or values.shape[1] != 3:
    raise ValueError(
        f'Ожидалось 3 столбца (G, BP-RP, parallax), получено: {values.shape}'
    )

if not np.isfinite(values).all():
    raise ValueError('В values есть пропущенные или бесконечные значения')

N = len(values)


# Проверка входных данных генератора
def validate_input(values, size):
    a = np.asarray(values, dtype=float)

    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(
            'values должен быть непустым одномерным массивом конечных чисел'
        )

    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 0
    ):
        raise ValueError('size должен быть неотрицательным целым числом')

    return a


# Генератор 1: метод обратной функции распределения
def empirical_inverse_sample(values, size, rng):
    a = validate_input(values, size)
    ordered = np.sort(a)

    positions = np.floor(
        rng.random(size) * len(ordered)
    ).astype(np.int64)

    return ordered[positions]


# Генератор 2: метод суперпозиции
def empirical_superposition_sample(values, size, rng):
    a = validate_input(values, size)

    pivot = np.median(a)
    left = a[a <= pivot]
    right = a[a > pivot]

    if len(right) == 0:
        return empirical_inverse_sample(left, size, rng)

    choose_right = rng.random(size) >= len(left) / len(a)

    result = np.empty(size, dtype=float)

    result[~choose_right] = empirical_inverse_sample(
        left, int((~choose_right).sum()), rng
    )

    result[choose_right] = empirical_inverse_sample(
        right, int(choose_right.sum()), rng
    )

    return result


# Генерация и сравнение результатов
generator_rows = []
generated_by_method = {}

methods = [
    ('Обратная CDF', empirical_inverse_sample),
    ('Суперпозиция', empirical_superposition_sample)
]

for feature_index, (key, label) in enumerate(
    zip(feature_keys, feature_labels)
):
    source = values[:, feature_index]

    for method_index, (method_name, method) in enumerate(methods):
        generated = method(
            source,
            10000,
            np.random.default_rng(
                SEED + 700 + feature_index * 10 + method_index
            )
        )

        generated_by_method[(key, method_name)] = generated

        generator_rows.append({
            'Характеристика': label,
            'Метод': method_name,
            'n': len(generated),
            'Среднее базы': source.mean(),
            'Среднее генератора': generated.mean(),
            'Медиана базы': np.median(source),
            'Медиана генератора': np.median(generated),
            's базы': source.std(ddof=1),
            's генератора': generated.std(ddof=1),
            'Q1 базы': np.quantile(source, .25),
            'Q1 генератора': np.quantile(generated, .25),
            'Q3 базы': np.quantile(source, .75),
            'Q3 генератора': np.quantile(generated, .75),
            'KS D': stats.ks_2samp(source, generated).statistic,
            'Wasserstein': stats.wasserstein_distance(source, generated)
        })

generator_comparison = pd.DataFrame(generator_rows)

generator_comparison.to_csv(
    OUT / 'generator_comparison_all_features.csv',
    index=False
)


# Отображение таблицы в общем стиле
styled_html_table(
    generator_comparison.round(6),
    title='Сравнение эмпирических генераторов',
    digits=6
)


# Главный генератор G и его распределение
main_generated = generated_by_method[('G', 'Обратная CDF')]

pd.DataFrame({
    'G_generated': main_generated
}).to_csv(OUT / 'generated_G.csv', index=False)


# Сравнение эмпирических функций распределения
fig, axes = plt.subplots(
    1, 3, figsize=(15, 4), constrained_layout=True
)

for j, key in enumerate(feature_keys):
    source = values[:, j]
    gen = generated_by_method[(key, 'Обратная CDF')]

    axes[j].step(
        np.sort(source),
        np.arange(1, N + 1) / N,
        where='post',
        label='База',
        color='#697586'
    )

    axes[j].step(
        np.sort(gen),
        np.arange(1, len(gen) + 1) / len(gen),
        where='post',
        label='Генератор',
        color='#527fb5'
    )

    axes[j].set(
        xlabel=feature_labels[j],
        ylabel='F(x)',
        title=feature_labels[j],
        ylim=(0, 1)
    )

    axes[j].legend(frameon=False, fontsize=8)
    axes[j].grid(alpha=.15)

fig.savefig(
    OUT / 'generator_cdf_comparison_all_features.png',
    bbox_inches='tight'
)

plt.show()


# Итоговый вывод
cosmic_card("""
<p>
Обе реализации воспроизводят одномерные эмпирические распределения.
Это воспроизведение конечной базы, а не прогноз свойств неизвестной звезды.
</p>
""", title='Вывод')

# %% [56]
# @title

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Берём небольшую базу, где вероятности легко проверить вручную
data = np.array([1, 2, 2, 3, 4, 4, 4, 5])
n = 10000
seed = 2026

# Метод обратной CDF
rng1 = np.random.default_rng(seed)
ordered = np.sort(data)
indices = (rng1.random(n) * len(data)).astype(int)
inverse = ordered[indices]

# Метод суперпозиции
rng2 = np.random.default_rng(seed)
median = np.median(data)
left = data[data <= median]
right = data[data > median]

choose_right = rng2.random(n) >= len(left) / len(data)
superposition = np.empty(n, dtype=int)

superposition[~choose_right] = rng2.choice(
    left, size=(~choose_right).sum(), replace=True
)
superposition[choose_right] = rng2.choice(
    right, size=choose_right.sum(), replace=True
)

# Сравниваем теоретические вероятности с результатами
unique, counts = np.unique(data, return_counts=True)
theory = counts / len(data)

inverse_freq = np.array([
    np.mean(inverse == x) for x in unique
])
superposition_freq = np.array([
    np.mean(superposition == x) for x in unique
])

table = pd.DataFrame({
    'Значение': unique,
    'Теория': theory,
    'Обратная CDF': inverse_freq,
    'Суперпозиция': superposition_freq
})

# Оформление таблицы
styled_html_table(
    table.round(4),
    title='Сравнение теоретических и эмпирических вероятностей',
    digits=4
)

# Строим график в исходных цветах лабораторной — без изменений
x = np.arange(len(unique))
width = 0.26

fig, ax = plt.subplots(figsize=(10, 4.5))

ax.bar(x - width, theory, width,
       label='Теория', color='#697586')
ax.bar(x, inverse_freq, width,
       label='Обратная CDF', color='#527fb5')
ax.bar(x + width, superposition_freq, width,
       label='Суперпозиция', color='#A78BFA')

ax.set_xticks(x)
ax.set_xticklabels(unique)
ax.set_xlabel('Значение')
ax.set_ylabel('Относительная частота')
ax.set_title('Сравнение двух методов генерации')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.15)

plt.tight_layout()
plt.show()

# Результаты сравнения
inverse_deviation = np.max(np.abs(inverse_freq - theory))
superposition_deviation = np.max(np.abs(superposition_freq - theory))

cosmic_card(f"""
<div style="font-size:17px;line-height:1.9;color:#E4E8FF;">
    <div style="margin-bottom:10px;">
        Обратная CDF:
        <span style="color:#8FE3C1;font-weight:700;">{inverse_deviation:.4f}</span>
    </div>
    <div>
        Суперпозиция:
        <span style="color:#8FE3C1;font-weight:700;">{superposition_deviation:.4f}</span>
    </div>
</div>
""", title='Максимальное отклонение от теории')

# %% [57]
#@title

cosmic_card("""
<h2 style="color:#C9B6FF;font-size:27px;margin:0 0 22px;">Вывод</h2>

<div style="
    margin:16px 0;
    padding:20px 22px;
    border-radius:13px;
    background:rgba(120,100,220,.13);
    border:1px solid rgba(180,160,255,.28);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">
    При генерации 10 000 значений оба метода должны давать частоты, близкие к теоретическим.
    Различия между ними возникают из-за случайных колебаний, а не из-за изменения исходного распределения.
</div>

<div style="
    margin:16px 0;
    padding:20px 22px;
    border-radius:13px;
    background:rgba(100,190,255,.13);
    border:1px solid rgba(140,210,255,.38);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">
    Метод обратной CDF и суперпозиция при выбранном разбиении по медиане воспроизводят одно и то же эмпирическое распределение.
</div>

<div style="
    margin:16px 0 0;
    padding:20px 22px;
    border-radius:13px;
    background:rgba(120,100,220,.12);
    border:1px solid rgba(180,160,255,.20);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">
    Для нашей задачи обратная CDF проще, поскольку суперпозиция добавляет этап выбора группы, но не даёт нового распределения.
</div>
""")

# %% [58]
# @title

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

data = np.array([1, 2, 2, 3, 4, 4, 4, 5], dtype=float)
n = 100_000
seeds = [42, 2026, 12345]

def inverse_test(data, n, rng):
    ordered = np.sort(data)
    indices = (rng.random(n) * len(ordered)).astype(int)
    return ordered[indices]

def superposition_test(data, n, rng):
    pivot = np.median(data)
    left = data[data <= pivot]
    right = data[data > pivot]

    if len(right) == 0:
        return inverse_test(left, n, rng)

    choose_right = rng.random(n) >= len(left) / len(data)
    result = np.empty(n)
    result[~choose_right] = inverse_test(left, (~choose_right).sum(), rng)
    result[choose_right] = inverse_test(right, choose_right.sum(), rng)
    return result

methods = {
    'Обратная CDF': inverse_test,
    'Суперпозиция': superposition_test
}

unique, counts = np.unique(data, return_counts=True)
theory = counts / len(data)

rows = []

for seed in seeds:
    for name, method in methods.items():
        times = []
        errors = []

        for repeat in range(5):
            rng = np.random.default_rng(seed + repeat)

            start = perf_counter()
            generated = method(data, n, rng)
            times.append((perf_counter() - start) * 1000)

            frequencies = np.array([
                np.mean(generated == x) for x in unique
            ])
            errors.append(np.max(np.abs(frequencies - theory)))

        rows.append({
            'Метод': name,
            'Seed': seed,
            'Время, мс': np.median(times),
            'Макс. отклонение': np.mean(errors)
        })

results = pd.DataFrame(rows)

# Оформленная таблица результатов
styled_html_table(
    results.round(4),
    title='Скорость генераторов и точность',
    digits=4
)

# Сравнение скорости — исходный график без изменений
speed = results.groupby('Метод')['Время, мс'].mean()

plt.figure(figsize=(8, 4))
plt.bar(speed.index, speed.values)
plt.ylabel('Время генерации, мс')
plt.title('Сравнение скорости двух методов')
plt.tight_layout()
plt.show()

# Воспроизводимость последовательностей — исходные проверки
a = inverse_test(data, 12, np.random.default_rng(2026))
b = inverse_test(data, 12, np.random.default_rng(2026))
c = inverse_test(data, 12, np.random.default_rng(2027))

# Красивое оформление проверки воспроизводимости
def format_array(arr):
    return '[' + ', '.join(f'{x:g}' for x in arr) + ']'

same_seed = np.array_equal(a, b)
different_seed = np.array_equal(a, c)

cosmic_card(f"""
<div style="color:#E4E8FF;font-size:16px;line-height:1.8;">

    <div style="
        padding:14px 16px;
        margin:12px 0;
        border-radius:12px;
        background:rgba(120,100,220,.13);
        border:1px solid rgba(180,160,255,.28);
    ">
        <div style="font-weight:700;color:#C9B6FF;margin-bottom:8px;">
            Seed 2026 — первый запуск
        </div>
        <code style="font-size:13px;overflow-wrap:anywhere;color:#E4E8FF;">
            {format_array(a)}
        </code>
    </div>

    <div style="
        padding:14px 16px;
        margin:12px 0;
        border-radius:12px;
        background:rgba(100,190,255,.13);
        border:1px solid rgba(140,210,255,.38);
    ">
        <div style="font-weight:700;color:#A8DEFF;margin-bottom:8px;">
            Seed 2026 — второй запуск
        </div>
        <code style="font-size:13px;overflow-wrap:anywhere;color:#E4E8FF;">
            {format_array(b)}
        </code>
    </div>

    <div style="
        padding:14px 16px;
        margin:12px 0;
        border-radius:12px;
        background:rgba(65,120,220,.16);
        border:1px solid rgba(125,165,255,.38);
    ">
        <div style="font-weight:700;color:#B8CFFF;margin-bottom:8px;">
            Seed 2027
        </div>
        <code style="font-size:13px;overflow-wrap:anywhere;color:#E4E8FF;">
            {format_array(c)}
        </code>
    </div>

    <div style="
        font-weight:700;
        color:#C9B6FF;
        font-size:18px;
        margin:20px 0 12px;
    ">
        Результаты проверки
    </div>

    <div style="display:flex;gap:12px;flex-wrap:wrap;">

        <div style="
            flex:1;
            min-width:190px;
            padding:14px;
            border-radius:12px;
            background:rgba(120,100,220,.13);
            border:1px solid rgba(180,160,255,.28);
        ">
            <div style="color:#C9B6FF;font-size:14px;">
                Одинаковый seed
            </div>
            <div style="font-size:18px;font-weight:700;color:#8FE3C1;margin-top:6px;">
                {'Совпадают' if same_seed else 'Не совпадают'}
            </div>
            <div style="font-size:14px;margin-top:4px;">
                {str(same_seed)}
            </div>
        </div>

        <div style="
            flex:1;
            min-width:190px;
            padding:14px;
            border-radius:12px;
            background:rgba(100,190,255,.13);
            border:1px solid rgba(140,210,255,.38);
        ">
            <div style="color:#A8DEFF;font-size:14px;">
                Разный seed
            </div>
            <div style="font-size:18px;font-weight:700;color:#8FE3C1;margin-top:6px;">
                {'Совпадают' if different_seed else 'Не совпадают'}
            </div>
            <div style="font-size:14px;margin-top:4px;">
                {str(different_seed)}
            </div>
        </div>

    </div>
</div>
""", title='Проверка воспроизводимости генератора')

# %% [59]
#@title

cosmic_card("""
<h2 style="margin:0 0 22px;color:#C9B6FF;font-size:27px;line-height:1.4;">
Сравнение скорости и воспроизводимости генераторов
</h2>

<p style="margin:0 0 18px;">
По результатам сравнения двух методов при одинаковом объёме генерации (100 000 значений)
<strong style="color:#8FE3C1;">обратная CDF</strong> оказался быстрее.
Для каждого метода использовали три значения <code style="color:#A8DEFF;background:rgba(129,140,248,.13);padding:2px 6px;border-radius:5px;">seed</code>
и пять повторений. Оба алгоритма дали небольшие отклонения от теоретических вероятностей.
</p>

<div style="margin:26px 0 12px;padding:0 0 8px;border-bottom:1px solid rgba(165,180,252,.3);">
<h3 style="margin:0;color:#A8DEFF;font-size:21px;line-height:1.5;">
Влияние начального значения генератора
</h3>
</div>

<p style="margin:0 0 18px;">
При одинаковом
<code style="color:#A8DEFF;background:rgba(129,140,248,.13);padding:3px 7px;border-radius:5px;">seed = 2026</code>
полученные последовательности полностью совпали. При изменении
<code style="color:#A8DEFF;background:rgba(129,140,248,.13);padding:3px 7px;border-radius:5px;">seed</code>
последовательность изменилась.
</p>

<div style="
    margin:24px 0;
    padding:20px 22px;
    border-radius:14px;
    background:rgba(100,190,255,.13);
    border:1px solid rgba(140,210,255,.38);
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">
<p style="margin:0;">
Это показывает, что псевдослучайная последовательность может иметь правильное статистическое распределение и при этом оставаться воспроизводимой, если известно начальное состояние генератора.
</p>
</div>

<p style="margin:0;">
Однако воспроизводимость при известном
<code style="color:#A8DEFF;background:rgba(129,140,248,.13);padding:3px 7px;border-radius:5px;">seed</code>
сама по себе ещё не доказывает возможность восстановления неизвестного начального состояния.
</p>
""")

# %% [60]
# @title

import numpy as np
import pandas as pd

# Учебный эксперимент: seed выбирается из небольшого диапазона
secret_seed = 137
possible_seeds = range(1000)

# Наблюдаем первые 5 чисел генератора PCG64
observed = np.random.Generator(
    np.random.PCG64(secret_seed)
).integers(0, 1000000, size=5)

cosmic_card(f"""
<div style="
    padding:16px;
    border-radius:13px;
    background:rgba(65,120,220,.16);
    border:1px solid rgba(125,165,255,.38);
">
    <div style="
        font-size:18px;
        font-weight:700;
        color:#A8DEFF;
        margin-bottom:12px;
    ">
        Наблюдаемая последовательность
    </div>
    <div style="
        font-size:14px;
        color:#B8CFFF;
        margin-bottom:8px;
    ">
        Первые 5 чисел генератора PCG64
    </div>
    <div style="
        font-size:20px;
        font-weight:700;
        color:#B8DFFF;
        overflow-wrap:anywhere;
        line-height:1.7;
    ">
        {np.array2string(observed, separator=', ')}
    </div>
</div>
""")

# Проверяем возможные seed и сравниваем последовательности
matches = []

for candidate in possible_seeds:
    test = np.random.Generator(
        np.random.PCG64(candidate)
    ).integers(0, 1000000, size=5)

    if np.array_equal(test, observed):
        matches.append(candidate)

# Красиво оформленная таблица подходящих seed
matches_df = pd.DataFrame({
    "Подходящие seed": matches
})

if matches:
    styled_html_table(
        matches_df,
        title='Результат поиска seed'
    )
else:
    cosmic_card(
        "<p>Подходящие seed не найдены.</p>",
        title='Результат поиска seed'
    )

found_seed = matches[0] if matches else "не найден"

cosmic_card(f"""
<div style="display:flex;gap:12px;flex-wrap:wrap;">

    <div style="
        flex:1;
        min-width:180px;
        padding:15px;
        border-radius:12px;
        background:rgba(120,100,220,.13);
        border:1px solid rgba(180,160,255,.28);
    ">
        <div style="font-size:14px;color:#C9B6FF;">
            Исходный seed:
        </div>
        <div style="
            font-size:21px;
            font-weight:700;
            color:#D8CCFF;
            margin-top:5px;
        ">
            {secret_seed}
        </div>
    </div>

    <div style="
        flex:1;
        min-width:180px;
        padding:15px;
        border-radius:12px;
        background:rgba(100,190,255,.13);
        border:1px solid rgba(140,210,255,.38);
    ">
        <div style="font-size:14px;color:#A8DEFF;">
            Найденный seed:
        </div>
        <div style="
            font-size:21px;
            font-weight:700;
            color:#B8DFFF;
            margin-top:5px;
        ">
            {found_seed}
        </div>
    </div>

</div>
""", title='Сравнение исходного и найденного seed')

# Проверим, что найденный seed позволяет воспроизвести
# следующее значение в той же последовательности
if matches:
    original = np.random.Generator(np.random.PCG64(secret_seed))
    recovered = np.random.Generator(np.random.PCG64(matches[0]))

    original.integers(0, 1000000, size=5)
    recovered.integers(0, 1000000, size=5)

    next_original = original.integers(0, 1000000)
    next_recovered = recovered.integers(0, 1000000)

    is_equal = next_original == next_recovered

    cosmic_card(f"""
    <div style="
        margin:10px 0;
        padding:14px 16px;
        border-radius:12px;
        background:rgba(120,100,220,.13);
        border:1px solid rgba(180,160,255,.28);
    ">
        <div style="color:#C9B6FF;font-size:14px;margin-bottom:5px;">
            Следующее исходное число:
        </div>
        <div style="
            font-size:20px;
            font-weight:700;
            color:#D8CCFF;
        ">
            {next_original}
        </div>
    </div>

    <div style="
        margin:10px 0;
        padding:14px 16px;
        border-radius:12px;
        background:rgba(100,190,255,.13);
        border:1px solid rgba(140,210,255,.38);
    ">
        <div style="color:#A8DEFF;font-size:14px;margin-bottom:5px;">
            Следующее предсказанное число:
        </div>
        <div style="
            font-size:20px;
            font-weight:700;
            color:#B8DFFF;
        ">
            {next_recovered}
        </div>
    </div>

    <div style="
        margin-top:14px;
        padding:14px 16px;
        border-radius:12px;
        background:rgba(120,100,220,.12);
        border:1px solid rgba(180,160,255,.25);
    ">
        <div style="color:#C9B6FF;font-size:14px;">
            Совпадение:
        </div>
        <div style="
            font-size:20px;
            font-weight:700;
            color:#D8CCFF;
            margin-top:5px;
        ">
            {is_equal}
        </div>
    </div>
    """, title='Проверка предсказания следующего числа')

# %% [61]
# @title

cosmic_card("""
<div style="
    color:#E4E8FF;
    font-size:17px;
    line-height:1.9;
">

<h2 style="
    margin:0 0 22px;
    color:#C9B6FF;
    font-size:27px;
    line-height:1.4;
">
Исследование предсказуемости генератора
</h2>

<p style="margin:0 0 18px;">
Для проверки возможности восстановления начального значения генератора при известной части последовательности выбрали PCG64 и ограничили возможные значения
<code style="color:#B8CFFF;background:rgba(100,150,255,.13);padding:3px 7px;border-radius:5px;">seed</code> диапазоном от 0 до 999. Затем сгенерировали пять чисел и попытались найти начальное значение перебором всех вариантов.
</p>

<div style="
    margin:24px 0;
    padding:20px 22px;
    border-radius:14px;
    background:rgba(100,190,255,.13);
    border:1px solid rgba(140,210,255,.38);
">
<p style="margin:0;color:#DCEBFF;">
В результате удалось восстановить исходный
<code style="color:#A8DEFF;background:rgba(100,190,255,.14);padding:3px 7px;border-radius:5px;">seed</code>
и правильно предсказать следующее число.
</p>
</div>

<p style="margin:0 0 18px;">
Получается, что статистически случайная на вид последовательность может быть предсказуемой, если начальное значение выбирается из небольшого набора.
</p>

<p style="margin:0 0 18px;">
При этом наш эксперимент не означает, что любое начальное состояние PCG64 можно так же легко восстановить. Мы специально создали условия, при которых полный перебор был возможен.
</p>

<div style="
    margin:24px 0 0;
    padding:20px 22px;
    border-radius:14px;
    background:rgba(120,100,220,.15);
    border:1px solid rgba(180,160,255,.32);
">
<p style="margin:0;color:#E4DCFF;">
Это объясняет, почему для защиты данных недостаточно получить числа с хорошим распределением. Важно также, насколько трудно предсказать внутреннее состояние генератора.
</p>
</div>

</div>
""", title=None)

# %% [62]
# @title

cosmic_card("""
<div style="
    color:#E4E8FF;
    font-size:16px;
    line-height:1.85;
">

    <div style="
        color:#C9B6FF;
        font-size:13px;
        font-weight:bold;
        letter-spacing:2px;
        text-transform:uppercase;
        margin-bottom:10px;
    ">Реальный случай</div>

    <h2 style="
        margin:0 0 24px 0;
        color:#E5DDFF;
        font-size:27px;
        line-height:1.35;
        font-weight:700;
    ">Пример реальной уязвимости: Debian OpenSSL {}</h2>

    <p style="margin:0 0 18px 0;">
        В 2008 году была обнаружена уязвимость
        <span style="color:#D8CAFF;font-weight:bold;">CVE-2008-0166</span>
        в генераторе случайных чисел OpenSSL, использовавшемся в Debian. Причина — ошибка при изменении исходного кода. Из-за нее генератор получал недостаточно случайных данных, поэтому количество возможных криптографических ключей сильно сократилось.
    </p>

    <div style="
        margin:24px 0;
        padding:22px 24px;
        border-radius:14px;
        background:rgba(139,111,207,.13);
        border:1px solid rgba(180,155,255,.32);
    ">
        <div style="
            color:#D5C5FF;
            font-size:17px;
            font-weight:bold;
            margin-bottom:10px;
        ">Связь с экспериментом</div>
        <p style="margin:0;">
            Наш эксперимент с небольшим диапазоном <code style="color:#A5B4FC;background:rgba(129,140,248,.13);padding:2px 6px;border-radius:5px;">seed</code> не повторяет саму ошибку Debian, но показывает похожую проблему: если число возможных начальных состояний слишком мало, их можно перебрать.
        </p>
    </div>

    <div style="
        margin:24px 0 0 0;
        padding:22px 24px;
        border-radius:14px;
        background:rgba(100,190,255,.12);
        border:1px solid rgba(140,210,255,.34);
    ">
        <div style="
            color:#A8DEFF;
            font-size:17px;
            font-weight:bold;
            margin-bottom:10px;
        ">Главный вывод</div>
        <p style="margin:0;color:#DCEBFF;">
            То есть правильное распределение случайных чисел не гарантирует безопасность. Для криптографии важно также, насколько трудно предсказать внутреннее состояние генератора.
        </p>
    </div>

    <div style="
        margin-top:28px;
        padding-top:20px;
        border-top:1px solid rgba(180,195,255,.23);
    ">
        <div style="
            color:#C9B6FF;
            font-size:13px;
            font-weight:bold;
            letter-spacing:1.5px;
            margin-bottom:8px;
        ">ИСТОЧНИК</div>

        <div style="
            color:#D8D9F5;
            font-size:14px;
            margin-bottom:8px;
        ">Debian Security Advisory DSA-1571-1, CVE-2008-0161</div>

        <a href="https://lists.debian.org/debian-security-announce/2008/msg00152.html"
           target="_blank"
           style="
               color:#A8DEFF;
               text-decoration:none;
               overflow-wrap:anywhere;
               font-size:14px;
           "
        >lists.debian.org — Debian Security Advisory ↗</a>
    </div>

</div>
""", title=None)

# %% [63]
# @title

cosmic_card("""
<div style="
    color:#E4E8FF;
    font-size:16px;
    line-height:1.85;
">

    <h2 style="
        color:#C9B6FF;
        font-size:27px;
        margin:0 0 20px;
    ">Квантовая случайность и защита данных</h2>

    <p>При изучении генераторов мы также рассмотрели их применение в криптографии.</p>

    <div style="
        margin:20px 0;
        padding:20px;
        border-radius:13px;
        background:rgba(120,100,220,.13);
        border:1px solid rgba(180,160,255,.30);
    ">
        <div style="
            font-size:18px;
            font-weight:bold;
            color:#D5C8FF;
            margin-bottom:8px;
        ">Генераторы случайных чисел</div>

        PCG64 подходит для статистического моделирования, но не предназначен для защиты данных,
        поскольку его последовательность определяется внутренним состоянием. Для криптографии
        используют специальные стойкие генераторы. Существуют также квантовые генераторы (QRNG),
        которые получают случайные числа на основе физических процессов.
    </div>

    <div style="
        margin:20px 0;
        padding:20px;
        border-radius:13px;
        background:rgba(65,150,230,.12);
        border:1px solid rgba(120,190,255,.30);
    ">
        <div style="
            font-size:18px;
            font-weight:bold;
            color:#A9DCFF;
            margin-bottom:8px;
        ">Постквантовая криптография</div>

        <div style="color:#DCEBFF;">
            Отдельное направление — постквантовая криптография (PQC). Её задача —
            защищать информацию с помощью алгоритмов, устойчивых к известным атакам,
            в том числе с использованием квантовых компьютеров.
        </div>
    </div>

    <div style="
        margin-top:26px;
        padding-top:18px;
        border-top:1px solid rgba(180,195,255,.22);
    ">
        <div style="
            font-size:15px;
            font-weight:bold;
            color:#C9B6FF;
            margin-bottom:8px;
        ">Источник</div>

        <a href="https://www.nist.gov/cybersecurity-and-privacy/what-post-quantum-cryptography"
           target="_blank"
           rel="noopener noreferrer"
           style="
               color:#A9DCFF;
               text-decoration:none;
               overflow-wrap:anywhere;
           "
        >NIST — Post-Quantum Cryptography ↗</a>
    </div>

</div>
""", title=None)

# %% [64]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
12. Автоматические проверки генераторов
</h2></div>
"""))

# %% [65]
# @title

# Таблица семи автоматических тестов
tests_df = pd.DataFrame([
    ["Размер", "Генератор возвращает столько значений, сколько мы запросили"],
    ["Воспроизводимость", "При одинаковом seed последовательности совпадают"],
    ["Нулевой размер", "При size = 0 возвращается пустой массив"],
    ["Одна точка", "Если в базе одно значение, генератор возвращает только его"],
    ["Исходные значения", "Все результаты принадлежат исходной базе и являются конечными числами"],
    ["Повторяющиеся значения", "Проверяем частоту значения, которое встречается в базе несколько раз"],
    ["Ошибочные аргументы", "Некорректные данные должны вызывать ValueError"]
], columns=["Проверка", "Что проверяем"])


# Основной текст и формула
cosmic_card("""
<div style="
    color:#E4E8FF;
    font-size:16px;
    line-height:1.85;
">

<h2 style="
    margin:0 0 18px;
    color:#C9B6FF;
    font-size:25px;
    line-height:1.4;
">
Для проверки двух генераторов мы использовали семь автоматических тестов.
</h2>

<p style="margin:0 0 22px;">
Они помогают убедиться, что функции правильно обрабатывают входные данные и воспроизводят ожидаемые свойства распределения.
</p>

</div>
""", title=None)

styled_html_table(
    tests_df,
    title='Семь автоматических тестов',
    digits=6
)

cosmic_card("""
<div style="
    color:#E4E8FF;
    font-size:16px;
    line-height:1.85;
">

<p style="margin:0 0 18px;">
Для проверки повторяющихся значений используем базу [1,1,3]. Теоретическая вероятность получить число 3 равна 1/3. При генерации 60 000 значений проверяем, что наблюдаемая частота не слишком сильно отклоняется от неё.
</p>

<p style="margin:22px 0 12px;">
Допустимое отклонение рассчитывается по формуле:
</p>

<div style="
    margin:15px 0 24px;
    padding:25px 14px;
    border-radius:14px;
    background:rgba(113,93,194,.18);
    border:1px solid rgba(190,174,255,.35);
    text-align:center;
    color:#E1D4FF;
">

<div style="
    display:inline-flex;
    align-items:center;
    gap:7px;
    font-family:'Times New Roman',serif;
    font-size:25px;
    white-space:nowrap;
">

<span style="font-size:30px;">|</span>

<span><i>p̂</i> − <span style="
    display:inline-flex;
    flex-direction:column;
    vertical-align:middle;
    text-align:center;
    font-size:19px;
    line-height:1.15;
"><span style="border-bottom:1px solid #e1d4ff;padding:0 7px 3px;">1</span><span style="padding-top:3px;">3</span></span></span>

<span style="font-size:30px;">|</span>

<span>&lt;</span>

<span>6</span>

<span style="display:inline-flex;align-items:center;">

<span style="font-size:47px;line-height:1;">√</span>

<span style="
    display:inline-flex;
    flex-direction:column;
    vertical-align:middle;
    text-align:center;
    font-size:18px;
    line-height:1.35;
    border-top:1px solid #e1d4ff;
    padding:4px 7px 0;
">

<span>
<span style="
    display:inline-flex;
    flex-direction:column;
    vertical-align:middle;
    text-align:center;
    font-size:16px;
    line-height:1.1;
"><span style="border-bottom:1px solid #e1d4ff;padding:0 5px 3px;">1</span><span style="padding-top:3px;">3</span></span>
<span style="padding:0 3px;">·</span>
<span style="
    display:inline-flex;
    flex-direction:column;
    vertical-align:middle;
    text-align:center;
    font-size:16px;
    line-height:1.1;
"><span style="border-bottom:1px solid #e1d4ff;padding:0 5px 3px;">2</span><span style="padding-top:3px;">3</span></span>
</span>

<span style="
    border-top:1px solid rgba(225,212,255,.6);
    margin-top:4px;
    padding-top:3px;
">60000</span>

</span>
</span>

</div>
</div>

<p style="margin:0 0 20px;">
Здесь p̂ — полученная относительная частота числа 3.
</p>

<div style="
    margin:22px 0;
    padding:17px 20px;
    border-radius:12px;
    background:rgba(105,157,220,.12);
    border:1px solid rgba(135,190,255,.3);
    color:#DCEBFF;
">
Выбран достаточно широкий допуск, чтобы тест не срабатывал из-за обычных случайных колебаний.
</div>

<p style="
    margin:22px 0 0;
    color:#D9DEF7;
">
Проверки показывают правильность работы функций в выбранных условиях, но не доказывают абсолютную случайность последовательностей или их криптографическую безопасность.
</p>

</div>
""", title=None)

# %% [66]
# @title

class GeneratorTests(unittest.TestCase):
    methods = (empirical_inverse_sample, empirical_superposition_sample)

    def test_size(self):
        for method in self.methods:
            self.assertEqual(
                method([1., 2., 5.], 123, np.random.default_rng(1)).shape,
                (123,)
            )

    def test_reproducibility(self):
        for method in self.methods:
            a = method([1., 2., 5.], 1000, np.random.default_rng(15))
            b = method([1., 2., 5.], 1000, np.random.default_rng(15))
            np.testing.assert_array_equal(a, b)

    def test_empty(self):
        for method in self.methods:
            self.assertEqual(
                method([1., 2.], 0, np.random.default_rng(1)).shape,
                (0,)
            )

    def test_singleton(self):
        for method in self.methods:
            np.testing.assert_array_equal(
                method([7.], 100, np.random.default_rng(2)),
                np.full(100, 7.)
            )

    def test_support(self):
        for method in self.methods:
            d = method([-3., 1., 1., 9.], 5000, np.random.default_rng(11))
            self.assertTrue(
                np.isfinite(d).all() and np.isin(d, [-3., 1., 9.]).all()
            )

    def test_duplicate_weights(self):
        for method in self.methods:
            d = method([1., 1., 3.], 60000, np.random.default_rng(31))
            self.assertLess(
                abs(np.mean(d == 3) - 1/3),
                6 * np.sqrt((1/3) * (2/3) / 60000)
            )

    def test_invalid(self):
        for method in self.methods:
            for values_bad in [[], [[1., 2.]], [np.nan], [np.inf], 5.]:
                with self.assertRaises(ValueError):
                    method(values_bad, 1, np.random.default_rng(4))

            for size_bad in [-1, 1.5, True]:
                with self.assertRaises(ValueError):
                    method([1., 2.], size_bad, np.random.default_rng(4))


suite = unittest.defaultTestLoader.loadTestsFromTestCase(GeneratorTests)
result = unittest.TestResult()
suite.run(result)

if not result.wasSuccessful():
    raise AssertionError(result.errors + result.failures)

test_results = pd.DataFrame({
    'Проверка': [
        'Размер',
        'Воспроизводимость',
        'Нулевой размер',
        'Одна точка',
        'Диапазон исходной базы',
        'Повторяющиеся значения',
        'Ошибочные аргументы'
    ],
    'Результат': ['Пройдена'] * 7
})

test_results.to_csv(OUT / 'generator_tests.csv', index=False)

# Отображение в едином HTML-стиле
styled_html_table(
    test_results,
    title='Результаты автотестов генераторов'
)

cosmic_card(
    """
    <div style="
        color: #DCEBFF;
        font-size: 16px;
        line-height: 1.85;
    ">
        <span style="
            color: #A8DEFF;
            font-weight: 600;
        ">Пройдено проверок:</span>
        <span style="
            color: #B8DFFF;
            font-weight: 700;
        ">7</span>
    </div>
    """,
    title='Итог тестирования'
)

print('Пройдено проверок:', result.testsRun)

# %% [67]
#@title

cosmic_card(
    """
    <div style="
        color: #E4E8FF;
        font-family: Arial, sans-serif;
        font-size: 16px;
        line-height: 1.85;
    ">

        <h2 style="
            margin: 0 0 22px;
            color: #C9B6FF;
            font-size: 25px;
        ">Учебные реализации двух дополнительных алгоритмов</h2>

        <p>
            Метод Неймана генерирует кандидата из простой плотности и принимает его с вероятностью, зависящей от отношения целевой плотности к огибающей.
        </p>

        <p>
            Метод отношений равномерных величин строит точку в специальной области и возвращает отношение координат.
        </p>

        <div style="
            margin: 24px 0 0;
            padding: 20px 22px;
            border-radius: 14px;
            background: rgba(120,100,220,.15);
            border: 1px solid rgba(180,160,255,.32);
            color: #D8CCFF;
        ">
            <p style="margin: 0;">
                Оба метода используют равномерные числа PCG64 и служат примером алгоритмов для N(0,1), а не генератора Gaia.
            </p>
        </div>

    </div>
    """,
    title=None
)

# %% [68]
# @title

def rejection_normal(size, seed):
    rng = np.random.default_rng(seed)
    out = []
    proposed = accepted = 0

    while sum(len(part) for part in out) < size:
        y = rng.laplace(size=4096)
        u = rng.random(4096)
        ok = u <= np.exp(-0.5 * (np.abs(y) - 1)**2)

        proposed += len(y)
        accepted += int(ok.sum())
        out.append(y[ok])

    return np.concatenate(out)[:size], accepted / proposed


def ratio_uniforms_normal(size, seed):
    rng = np.random.default_rng(seed)
    out = []
    proposed = accepted = 0

    vmax = np.sqrt(2 / np.e)

    while sum(len(part) for part in out) < size:
        u = np.maximum(
            rng.random(4096),
            np.nextafter(0., 1.)
        )
        v = rng.uniform(-vmax, vmax, 4096)
        y = v / u
        ok = 2 * np.log(u) <= -0.5 * y * y

        proposed += len(u)
        accepted += int(ok.sum())
        out.append(y[ok])

    return np.concatenate(out)[:size], accepted / proposed


demos = []

for name, fn, seed in [
    ('Нейман', rejection_normal, SEED + 900),
    ('Ratio of uniforms', ratio_uniforms_normal, SEED + 901)
]:
    generated, acceptance = fn(10000, seed)

    demos.append({
        'Метод': name,
        'Среднее': generated.mean(),
        's': generated.std(ddof=1),
        'KS D к N(0,1)': stats.kstest(generated, 'norm').statistic,
        'Доля принятых': acceptance
    })


algorithm_demos = pd.DataFrame(demos)

algorithm_demos.to_csv(
    OUT / 'additional_algorithm_demonstrations.csv',
    index=False
)


# Оформленная таблица результатов
algorithm_display = algorithm_demos.round(6)

styled_html_table(
    algorithm_display,
    title='Сравнение алгоритмов генерации стандартной нормали'
)

# %% [69]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
13. Итоговые выводы
</h2></div>
"""))

# %% [70]
#@title

cosmic_card(
    """
    <div id="conclusions" style="
        width: 100%;
        box-sizing: border-box;
        font-family: Arial, sans-serif;
        color: #E4E8FF;
        line-height: 1.85;
        font-size: 16px;
    ">

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">1</div>
            <div style="flex:1;min-width:0;">
                Мы исследовали яркость <strong style="color:#C9B6FF;">G</strong>, цвет <strong style="color:#C9B6FF;">BP − RP</strong> и измеренный параллакс объектов <strong style="color:#C9B6FF;">Gaia DR3</strong>. Случайность в нашем опыте задаётся равновероятным выбором объекта из рабочей базы: до выбора неизвестно, какие значения попадутся.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">2</div>
            <div style="flex:1;min-width:0;">
                Рабочую базу из <strong style="color:#C9B6FF;">10 000 строк</strong> использовали как конечную генеральную совокупность для вычислительных экспериментов. По ней можно проверить точность оценок, потому что характеристики всей базы нам известны. При этом её состав зависит от первоначального отбора и доступности измерений Gaia.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">3</div>
            <div style="flex:1;min-width:0;">
                Для каждой характеристики рассчитали описательные статистики, построили графики и выполнили нормализацию. <strong style="color:#C9B6FF;">Min–max</strong> и стандартизация помогают сравнивать масштабы, но сами по себе не делают распределение нормальным.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">4</div>
            <div style="flex:1;min-width:0;">
                Сравнили несколько моделей распределения. Меньший <strong style="color:#C9B6FF;">BIC</strong> показывает преимущество модели среди рассмотренных вариантов, но не гарантирует её согласия с данными. Эмпирическое распределение описывает случайный выбор из нашей конечной базы.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">5</div>
            <div style="flex:1;min-width:0;">
                Во всех трёх парах характеристик обнаружили статистически значимую, но слабую монотонную связь. Коэффициенты Спирмена составили примерно <strong style="color:#C9B6FF;">0.144, −0.093 и 0.129</strong>. Поэтому наличие связи не означает, что по одной характеристике можно точно предсказать другую.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">6</div>
            <div style="flex:1;min-width:0;">
                Повторные выборки показали, что с увеличением их размера оценки в целом становятся точнее. Например, при отборе без возвращения увеличение размера с <strong style="color:#C9B6FF;">20 до 500 объектов</strong> уменьшило RMSE среднего примерно в пять раз для всех трёх характеристик. Повторения позволяют увидеть разброс результатов, который нельзя оценить по одной выбранной группе.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 20px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">7</div>
            <div style="flex:1;min-width:0;">
                Отбор 500 самых ярких объектов изменил средний цвет с <strong style="color:#C9B6FF;">1.621 до 1.368 mag</strong>, а средний измеренный параллакс — с <strong style="color:#C9B6FF;">0.372 до 0.870 mas</strong>. Значит, такую группу нельзя использовать для описания средних характеристик всей базы без учёта смещения отбора.
            </div>
        </div>

        <div style="display:flex;gap:15px;margin:0 0 26px;align-items:flex-start;">
            <div style="flex:0 0 34px;height:34px;border-radius:10px;background:rgba(180,160,255,.16);border:1px solid rgba(180,160,255,.32);display:flex;align-items:center;justify-content:center;color:#C9B6FF;font-weight:bold;">8</div>
            <div style="flex:1;min-width:0;">
                Реализовали два способа генерации значений по эмпирическому распределению. Они воспроизводят одномерные распределения рабочей базы. Если генерировать три характеристики независимо, исходные связи между ними не сохранятся.
            </div>
        </div>

        <div style="
            padding:18px 22px;
            border-radius:14px;
            background:rgba(120,100,220,.15);
            border:1px solid rgba(180,160,255,.32);
            color:#D8CCFF;
        ">
            <p style="margin:0;">
                Следующая кодовая ячейка показывает численные результаты текущего запуска.
            </p>
        </div>

    </div>
    """,
    title='Выводы исследования'
)

# %% [71]
# @title

from IPython.display import display, HTML
import pandas as pd
import numpy as np
import html

# Единый стиль таблиц
def styled_html_table(df, title, digits=6):
    df = df.copy()

    def format_value(value):
        if pd.isna(value):
            return "NaN"

        if isinstance(value, (int, np.integer)):
            return str(value)

        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"

        return str(value)

    headers = "".join(
        f'<th style="padding:14px;text-align:left;'
        f'white-space:nowrap;">{html.escape(str(col))}</th>'
        for col in df.columns
    )

    rows = []
    for i, row in enumerate(df.itertuples(index=False, name=None)):
        background = "#F5F3FF" if i % 2 == 0 else "#FFFFFF"

        cells = "".join(
            '<td style="padding:12px;'
            'border-bottom:1px solid #E5E7EB;">'
            f'{html.escape(format_value(value))}</td>'
            for value in row
        )

        rows.append(
            f'<tr style="background:{background};">{cells}</tr>'
        )

    display(HTML(f"""
    <div style="font-family:Arial,sans-serif;margin:16px 0;">
      <div style="font-size:16px;font-weight:bold;color:#111827;
                  margin:0 0 10px 0;">
        {html.escape(title)}
      </div>
      <div style="overflow-x:auto;border:1px solid #D8D4F0;
                  border-radius:10px;">
        <table style="width:100%;border-collapse:separate;
                      border-spacing:0;font-size:14px;
                      color:#1F2937;">
          <thead>
            <tr style="background:#111827;color:#F8FAFC;">
              {headers}
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    </div>
    """))


# 1. ЧИСЛЕННЫЙ ИТОГ
summary_rows = []

for j, key in enumerate(feature_keys):
    a = values[:, j]

    summary_rows.append({
        "Характеристика": feature_labels[j],
        "N": N,
        "Среднее": a.mean(),
        "Медиана": np.median(a),
        "s": a.std(ddof=1),
        "Дисперсия": a.var(ddof=1)
    })

styled_html_table(
    pd.DataFrame(summary_rows),
    "ЧИСЛЕННЫЙ ИТОГ"
)


# 2. Зависимости
styled_html_table(
    dependence_table[
        ["Пара", "Spearman ρ", "p_Holm", "Решение"]
    ].round(6),
    "Зависимости"
)


# 3. Яркостный отбор
styled_html_table(
    selection_table[
        [
            "Характеристика",
            "Среднее базы",
            "Среднее ярких 500",
            "Разность",
            "p_Holm",
            "Решение"
        ]
    ].round(6),
    "Яркостный отбор"
)


# 4. Сравнение генераторов
styled_html_table(
    generator_comparison[
        [
            "Характеристика",
            "Метод",
            "Среднее базы",
            "Среднее генератора",
            "KS D",
            "Wasserstein"
        ]
    ].round(6),
    "Генератор"
)

# %% [72]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
14. Карта требований лабораторной работы
</h2></div>
"""))

# %% [73]
# @title

import pandas as pd

from IPython.display import display, HTML
import html

criteria = [
    (
        "Основное: датасет по направлению",
        "Раздел 1: Gaia DR3, источник данных и способ отбора"
    ),
    (
        "Основное: три случайные величины",
        "Разделы 3–5: G, BP − RP и измеренный параллакс"
    ),
    (
        "Основное: обоснование случайности",
        "Раздел 2: равновероятный выбор объекта из рабочей базы"
    ),
    (
        "Основное: описательная статистика",
        "Разделы 3–5: статистики, гистограммы, boxplot и ECDF"
    ),
    (
        "Основное: нормализация",
        "Разделы 3–5: min–max и стандартизация"
    ),
    (
        "Основное: исследование зависимости",
        "Раздел 6: Пирсон, Спирмен и перестановочная проверка"
    ),
    (
        "Основное: генеральная совокупность и выборки",
        "Раздел 7: рабочая база и меньшие случайные выборки"
    ),
    (
        "Выборки с возвращением и без возвращения",
        "Раздел 7: объяснение способов отбора и сравнение результатов"
    ),
    (
        "Основное: эмпирическое исследование точности оценок",
        "Раздел 7: смещение и RMSE при разных размерах выборки"
    ),
    (
        "Основное: статистические гипотезы",
        "Разделы 3–9: модели распределений, зависимости и яркостный отбор"
    ),
    (
        "Основное: математическая база распределений",
        "Разделы 3–5: модели; пояснения"
    ),
    (
        "Генерация значений по распределению",
        "Разделы 10–11: эмпирические генераторы"
    ),
    (
        "Обоснование алгоритмов генерации",
        "Раздел 11: обратный метод и суперпозиция"
    ),
    (
        "Сравнение исходных и сгенерированных данных",
        "Раздел 11: статистики, KS D и расстояние Вассерштейна"
    ),
    (
        "Доп.: другой алгоритм генерации",
        "Раздел 11: суперпозиция в дополнение к обратному методу"
    ),
    (
        "Доп.: автоматические проверки генератора",
        "Раздел 12: семь тестов для двух реализаций"
    ),
    (
        "Доп.: исследование дополнительных алгоритмов",
        "Раздел 12: принятие–отклонение и ratio of uniforms для N(0,1)"
    ),
    (
        "Доп.: рассмотрение разных распределений",
        "Разделы 3–5: нормальное, Лапласа и асимметричное нормальное"
    ),
    (
        "Доп.: приведение к стандартному нормальному",
        "Стандартизация выполнена; нормальность преобразованных данных этим не доказана"
    ),
    (
        "Доп.: табличное представление метрик",
        "Таблицы результатов и сохранение CSV"
    ),
    (
        "Доп.: список литературы",
        "Раздел 15: источники данных, методов и программных средств"
    ),
    (
        "Доп.: версионирование",
        "Подтверждается репозиторием GitHub и историей коммитов"
    ),
    (
        "Дополнительный эксперимент команды: смещение отбора",
        "Раздел 9: сравнение самых ярких объектов со случайными группами"
    ),
]

df_criteria = pd.DataFrame(
    [
        {
            "№": i + 1,
            "Требование": requirement,
            "Где выполнено": location
        }
        for i, (requirement, location) in enumerate(criteria)
    ]
)

styled_html_table(
    df_criteria,
    title="Критерии выполнения лабораторной работы"
)

cosmic_card(
    """
    <div style="
        font-family:Arial,sans-serif;
        color:#DCEBFF;
        line-height:1.85;
        padding:4px 2px;
    ">
        <strong style="color:#A8DEFF;">История версий:</strong>
        Коммиты создаются в GitHub.
    </div>
    """,
    title=None
)

# %% [74]
#@title
from IPython.display import display, HTML

display(HTML("""
<div style="background:#111827;color:#F8FAFC;padding:20px 24px;
border-left:5px solid #A78BFA;border-radius:10px;
font-family:Arial,sans-serif;box-sizing:border-box;">
<h2 style="color:#F8FAFC;margin:0;font-size:23px;font-weight:700;line-height:1.5;">
15. Литература и источники
</h2></div>
"""))

# %% [75]
# @title

params = {
    'seed': SEED,
    'rows': N,
    'source_sha256': PROVENANCE['raw_sha256'],
    'sampling_repeats': REPEATS,
    'permutation_repeats': B_PERM,
    'selection_repeats': SELECTION_REPEATS,
    'alpha': ALPHA,
    'generator_tests': result.testsRun,
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'scipy': __import__('scipy').__version__
}

(OUT / 'analysis_parameters.json').write_text(
    json.dumps(params, ensure_ascii=False, indent=2),
    encoding='utf-8'
)

df_params = pd.DataFrame(
    [
        {
            'Параметр': key,
            'Значение': value
        }
        for key, value in params.items()
    ]
)

styled_html_table(
    df_params,
    title="Параметры воспроизводимости"
)

# %% [76]
# @title

cosmic_card(
    """
    <div style="
        width:100%;
        box-sizing:border-box;
        padding:8px 4px;
        color:#E4E8FF;
        font-family:Arial,sans-serif;
        font-size:18px;
        line-height:1.9;
    ">

        <h2 style="
            color:#C9B6FF;
            font-size:27px;
            margin:0 0 22px;
        ">
            Проверка данных по исходному каталогу
        </h2>

        <p style="font-size:18px;line-height:1.9;">
            Для проверки выбрали три объекта из начала, середины и конца нашего CSV. Нашли их в официальном архиве Gaia по
            <code style="
                color:#B8DFFF;
                background:rgba(100,150,230,.13);
                padding:3px 7px;
                border-radius:5px;
                font-size:17px;
            ">source_id</code>
            и сравнили яркость G, цвет BP − RP и параллакс. Все девять значений совпали.
        </p>

        <div style="
            display:grid;
            grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
            gap:14px;
            margin:24px 0;
        ">

            <div style="
                padding:19px 15px;
                text-align:center;
                border-radius:12px;
                background:rgba(120,100,220,.13);
                border:1px solid rgba(180,160,255,.28);
            ">
                <div style="
                    font-size:29px;
                    color:#D5C8FF;
                    font-weight:bold;
                ">3</div>
                <div style="font-size:16px;color:#E4E0FF;">
                    проверенных объекта
                </div>
            </div>

            <div style="
                padding:19px 15px;
                text-align:center;
                border-radius:12px;
                background:rgba(65,150,230,.13);
                border:1px solid rgba(120,190,255,.30);
            ">
                <div style="
                    font-size:29px;
                    color:#A9DCFF;
                    font-weight:bold;
                ">9</div>
                <div style="font-size:16px;color:#DCEBFF;">
                    сверенных значений
                </div>
            </div>

            <div style="
                padding:19px 15px;
                text-align:center;
                border-radius:12px;
                background:rgba(65,190,160,.11);
                border:1px solid rgba(100,220,190,.28);
            ">
                <div style="
                    font-size:29px;
                    color:#9CEBD5;
                    font-weight:bold;
                ">0</div>
                <div style="font-size:16px;color:#D9F5EC;">
                    обнаруженных расхождений
                </div>
            </div>

        </div>

        <div style="
            padding:20px 22px;
            border-radius:12px;
            background:rgba(65,150,230,.13);
            border:1px solid rgba(120,195,255,.30);
            color:#DCEBFF;
            font-size:18px;
            line-height:1.9;
        ">
            Ошибок переноса в проверенных записях не обнаружили; остальные строки таким способом не проверяли.
        </div>

    </div>
    """,
    title=None
)

# %% [77]
# @title Математическая база распределений

distribution_theory_content = """
<div style="
    color:#72E5C2;
    font-size:13px;
    letter-spacing:2px;
    font-weight:bold;
    margin-bottom:12px;
">
    КТВД · ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ
</div>

<p style="color:#DDE5FF;font-size:16px;line-height:1.7;">
    В рамках дополнительного задания по курсу КТвД проводится
    статистическое исследование экзопланет на основе данных
    NASA Exoplanet Archive.
</p>

<p style="color:#DDE5FF;font-size:16px;line-height:1.7;">
    В исследовании рассматриваются две случайные величины:
    масса экзопланеты и период её обращения вокруг звезды.
    Цель анализа — изучить их распределения, статистические
    характеристики и возможную зависимость между ними.
</p>

<div style="margin:22px 0;padding:20px;border-radius:14px;
background:rgba(145,175,255,.09);border:1px solid rgba(145,175,255,.25);">
    <div style="color:#91AFFF;font-size:18px;font-weight:bold;margin-bottom:12px;">
        01 · Математическое ожидание
    </div>
    <p style="color:#DDE5FF;line-height:1.7;">
        Для дискретной случайной величины математическое ожидание
        определяется как сумма произведений возможных значений
        на их вероятности:
    </p>
    <div style="color:#91AFFF;font-size:25px;text-align:center;padding:15px 0;">
        E(X) = ∑ xᵢ pᵢ
    </div>
    <p style="color:#BFCBEA;font-size:14px;line-height:1.6;margin-bottom:0;">
        <b>xᵢ</b> — возможные значения случайной величины,
        <b>pᵢ</b> — вероятности этих значений.
    </p>
</div>

<div style="margin-bottom:22px;padding:20px;border-radius:14px;
background:rgba(201,182,255,.09);border:1px solid rgba(201,182,255,.25);">
    <div style="color:#C9B6FF;font-size:18px;font-weight:bold;margin-bottom:12px;">
        02 · Дисперсия
    </div>
    <p style="color:#DDE5FF;line-height:1.7;">
        Дисперсия характеризует разброс значений случайной величины
        относительно её математического ожидания:
    </p>
    <div style="color:#C9B6FF;font-size:24px;text-align:center;padding:15px 0;">
        D(X) = E[(X − E(X))²]
    </div>
    <p style="color:#BFCBEA;font-size:14px;line-height:1.6;margin-bottom:0;">
        Чем больше дисперсия, тем сильнее значения разбросаны
        относительно математического ожидания.
    </p>
</div>

<div style="padding:20px;border-radius:14px;
background:rgba(114,229,192,.08);border:1px solid rgba(114,229,192,.25);">
    <div style="color:#72E5C2;font-size:18px;font-weight:bold;margin-bottom:12px;">
        03 · Непрерывные случайные величины
    </div>
    <p style="color:#DDE5FF;line-height:1.7;">
        Для непрерывной случайной величины вероятности задаются
        плотностью распределения <b>f(x)</b>. Математическое ожидание
        и дисперсия в этом случае определяются с помощью интегралов:
    </p>
    <div style="color:#72E5C2;font-size:20px;text-align:center;padding:12px 0;line-height:1.8;">
        E(X) = ∫ x f(x) dx
        <br>
        D(X) = ∫ (x − E(X))² f(x) dx
    </div>
</div>

<div style="margin-top:22px;padding:18px 20px;border-radius:14px;
background:rgba(255,209,138,.08);border:1px solid rgba(255,209,138,.25);">
    <div style="color:#FFD18A;font-size:16px;font-weight:bold;margin-bottom:8px;">
        Источник данных
    </div>
    <p style="color:#DDE5FF;margin-bottom:0;">
        NASA Exoplanet Archive · таблица PSCompPars
    </p>
</div>
"""

cosmic_card(
    distribution_theory_content,
    title="Математическая база распределений"
)

# %% [78]
# @title Обоснование случайной природы исследуемых величин

if "df" not in globals():
    raise NameError("Сначала выполни загрузку данных NASA Exoplanet Archive.")

n_exoplanets = len(df)

if n_exoplanets == 0:
    raise ValueError("В датасете нет наблюдений для исследования.")

random_variables_content = f"""
<div style="
    color:#72E5C2;
    font-size:13px;
    letter-spacing:2px;
    font-weight:bold;
    margin-bottom:12px;
">
    STATISTICS · RANDOM VARIABLES
</div>

<p style="color:#DDE5FF;font-size:16px;line-height:1.7;">
    Для обоснования случайной природы исследуемых величин
    рассмотрим случайный эксперимент: выбор одной экзопланеты
    из сформированной выборки NASA Exoplanet Archive.
    Каждая из {n_exoplanets} экзопланет рассматривается как
    равновероятный объект выбора.
</p>

<div style="
    padding:20px;
    margin:20px 0;
    border-radius:14px;
    background:rgba(145,175,255,.09);
    border:1px solid rgba(145,175,255,.25);
">
    <div style="
        color:#91AFFF;
        font-size:18px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        01 · Случайная величина — масса экзопланеты
    </div>

    <p style="color:#DDE5FF;line-height:1.7;">
        Пусть X — масса случайно выбранной экзопланеты,
        выраженная в массах Юпитера.
        Значение X определяется тем, какая именно планета
        была выбрана из выборки.
    </p>

    <div style="
        color:#91AFFF;
        font-size:22px;
        text-align:center;
        padding:12px 0;
    ">
        X : Ω → ℝ
    </div>

    <p style="color:#BFCBEA;font-size:14px;line-height:1.6;margin-bottom:0;">
        Возможные значения X — массы экзопланет из датасета.
        Поскольку результат выбора заранее неизвестен,
        масса выбранного объекта рассматривается как случайная величина.
    </p>
</div>

<div style="
    padding:20px;
    margin:20px 0;
    border-radius:14px;
    background:rgba(114,229,192,.08);
    border:1px solid rgba(114,229,192,.25);
">
    <div style="
        color:#72E5C2;
        font-size:18px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        02 · Случайная величина — период обращения
    </div>

    <p style="color:#DDE5FF;line-height:1.7;">
        Пусть Y — период обращения той же случайно выбранной
        экзопланеты вокруг её звезды, выраженный в земных сутках.
        Его значение также зависит от результата случайного выбора.
    </p>

    <div style="
        color:#72E5C2;
        font-size:22px;
        text-align:center;
        padding:12px 0;
    ">
        Y : Ω → ℝ
    </div>

    <p style="color:#BFCBEA;font-size:14px;line-height:1.6;margin-bottom:0;">
        Возможные значения Y — периоды обращения планет
        из исследуемого датасета. Поэтому Y также является
        случайной величиной в рамках выбранного эксперимента.
    </p>
</div>

<div style="
    padding:20px;
    margin:20px 0;
    border-radius:14px;
    background:rgba(201,182,255,.09);
    border:1px solid rgba(201,182,255,.25);
">
    <div style="
        color:#C9B6FF;
        font-size:18px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        03 · Эмпирические распределения
    </div>

    <p style="color:#DDE5FF;line-height:1.7;">
        При равновероятном выборе одной планеты эмпирическая
        вероятность каждого значения определяется частотой его
        появления в выборке:
    </p>

    <div style="
        color:#C9B6FF;
        font-size:22px;
        text-align:center;
        padding:12px 0;
    ">
        P(X = x) = nₓ / N
    </div>

    <p style="color:#BFCBEA;font-size:14px;line-height:1.6;">
        Здесь nₓ — количество планет с массой x,
        N — общее количество планет в выборке.
        Аналогично определяется распределение периода обращения Y.
    </p>
</div>

<div style="
    padding:20px;
    border-radius:14px;
    background:rgba(255,209,138,.08);
    border:1px solid rgba(255,209,138,.25);
">
    <div style="
        color:#FFD18A;
        font-size:17px;
        font-weight:bold;
        margin-bottom:10px;
    ">
        Вывод
    </div>

    <p style="color:#DDE5FF;line-height:1.7;margin-bottom:0;">
        Таким образом, масса и период обращения являются
        случайными величинами относительно случайного выбора
        экзопланеты из исследуемой совокупности.
        Каждому выбранному объекту соответствует пара значений
        (X, Y). Их эмпирические распределения можно исследовать
        с помощью методов математической статистики.
    </p>
</div>

<p style="
    color:#AFC0E8;
    font-size:13px;
    line-height:1.6;
    margin-top:18px;
">
    Обоснование относится к сформированной выборке NASA Exoplanet Archive:
    оно не означает, что масса и период обращения всех существующих
    во Вселенной планет имеют такое же распределение.
</p>
"""

cosmic_card(
    random_variables_content,
    title="Обоснование случайной природы исследуемых величин"
)

# %% [79]
# @title Цветовая палитра исследования экзопланет

COLORS = {
    "mass": "#91AFFF",
    "period": "#72E5C2",
    "accent": "#C9B6FF",
    "pink": "#FF91C8",
    "gold": "#FFD18A",
}

# %% [80]
# @title Загрузка данных NASA Exoplanet Archive и EDA

from urllib.parse import quote

query = """
SELECT pl_name, pl_bmassj, pl_bmassprov, pl_orbper
FROM pscomppars
WHERE pl_bmassj IS NOT NULL
AND pl_bmassj > 0
AND pl_orbper IS NOT NULL
AND pl_orbper > 0
"""

url = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    f"query={quote(query)}&format=csv"
)

df = pd.read_csv(url).rename(columns={
    "pl_name": "Название",
    "pl_bmassj": "Масса (массы Юпитера)",
    "pl_bmassprov": "Источник оценки массы",
    "pl_orbper": "Период обращения (сутки)"
})

df = df.dropna(subset=[
    "Масса (массы Юпитера)",
    "Период обращения (сутки)"
])

df = df[
    (df["Масса (массы Юпитера)"] > 0) &
    (df["Период обращения (сутки)"] > 0)
].reset_index(drop=True)

mass = df["Масса (массы Юпитера)"]
period = df["Период обращения (сутки)"]

log_mass = np.log10(mass)
log_period = np.log10(period)

info = pd.DataFrame({
    "Показатель": [
        "Количество планет",
        "Количество исследуемых величин",
        "Единица измерения массы",
        "Единица измерения периода",
        "Источник данных",
        "Таблица NASA"
    ],
    "Значение": [
        len(df),
        2,
        "Массы Юпитера",
        "Земные сутки",
        "NASA Exoplanet Archive",
        "PSCompPars"
    ]
})

styled_html_table(info, "ИНФОРМАЦИЯ О ВЫБОРКЕ", digits=0)
styled_html_table(df.head(10), "ПЕРВЫЕ 10 НАБЛЮДЕНИЙ")

description = (
    df[[
        "Масса (массы Юпитера)",
        "Период обращения (сутки)"
    ]]
    .describe()
    .T
    .reset_index()
    .rename(columns={"index": "Характеристика"})
)

styled_html_table(description, "ОПИСАТЕЛЬНАЯ СТАТИСТИКА")

# Графики EDA
fig, axes = plt.subplots(2, 2, figsize=(17, 12))

fig.suptitle(
    "РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ · EDA",
    fontsize=22,
    fontweight="bold",
    color="#D4C4FF",
    y=1.01
)

plots = [
    (mass, "Распределение массы экзопланет",
     "Масса (массы Юпитера)", COLORS["mass"]),
    (period, "Распределение периода обращения",
     "Период обращения (сутки)", COLORS["period"]),
    (log_mass, "Распределение логарифма массы",
     "log₁₀(массы Юпитера)", COLORS["mass"]),
    (log_period, "Распределение логарифма периода",
     "log₁₀(периода в сутках)", COLORS["period"])
]

for ax, (values, title, xlabel, color) in zip(axes.flat, plots):
    ax.hist(
        values,
        bins=50,
        color=color,
        edgecolor="#E8EDFF",
        linewidth=0.5,
        alpha=0.9
    )
    ax.set_title(title, color="#D4C4FF", pad=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Количество планет")
    ax.grid(axis="y")

for ax in axes.flat:
    ax.tick_params(colors="#D5DCFA", labelsize=11)
    ax.xaxis.label.set_color("#DCE5FF")
    ax.yaxis.label.set_color("#DCE5FF")
    ax.spines["bottom"].set_color("#65749B")
    ax.spines["left"].set_color("#65749B")
    ax.grid(
        axis="y",
        color="#526184",
        linestyle="--",
        alpha=0.35
    )

plt.tight_layout()
plt.show()

cosmic_card("""
<p>
Гистограммы позволяют сравнить распределения массы
и периода обращения экзопланет.
</p>
<p>
Логарифмические гистограммы помогают рассмотреть данные
с большим диапазоном значений. Логарифмирование само по себе
не гарантирует нормальность распределения.
</p>
""", title="Вывод по EDA")

df.to_csv("exoplanets_mass_period.csv", index=False)

cosmic_card(f"""
<div style="text-align:center;">
    <div style="color:{COLORS['period']};font-size:21px;font-weight:bold;">
        ✓ ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ
    </div>
    <div style="font-size:36px;color:#FFFFFF;font-weight:bold;">
        {len(df):,}
    </div>
    <div style="color:#AFC0E8;">
        экзопланет в исследуемой выборке
    </div>
    <div style="margin-top:20px;color:{COLORS['accent']};">
        Файл exoplanets_mass_period.csv сохранён
    </div>
</div>
""", title="Загрузка завершена")

# %% [81]
# @title Нормализация и стандартизация данных

from sklearn.preprocessing import MinMaxScaler, StandardScaler

features = pd.DataFrame({
    "Логарифм массы": log_mass,
    "Логарифм периода": log_period
})

minmax_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

minmax_values = minmax_scaler.fit_transform(features)
standard_values = standard_scaler.fit_transform(features)

normalized = pd.DataFrame({
    "Логарифм массы": features["Логарифм массы"].values,
    "Масса Min-Max": minmax_values[:, 0],
    "Масса Z-score": standard_values[:, 0],
    "Логарифм периода": features["Логарифм периода"].values,
    "Период Min-Max": minmax_values[:, 1],
    "Период Z-score": standard_values[:, 1]
})

styled_html_table(
    normalized.head(10),
    "ПРИМЕР НОРМАЛИЗОВАННЫХ ДАННЫХ",
    digits=5
)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

plots = [
    (minmax_values[:, 0], "Масса · Min-Max", COLORS["mass"]),
    (standard_values[:, 0], "Масса · Z-score", COLORS["accent"]),
    (minmax_values[:, 1], "Период · Min-Max", COLORS["period"]),
    (standard_values[:, 1], "Период · Z-score", COLORS["gold"])
]

for ax, (values, title, color) in zip(axes.flat, plots):
    ax.hist(
        values,
        bins=45,
        color=color,
        edgecolor="#E8EDFF",
        linewidth=0.4,
        alpha=0.9
    )
    ax.set_title(title, color="#D4C4FF", fontweight="bold")
    ax.set_ylabel("Количество планет")
    ax.grid(axis="y")

for ax in axes.flat:
    ax.tick_params(colors="#D5DCFA")
    ax.xaxis.label.set_color("#DCE5FF")
    ax.yaxis.label.set_color("#DCE5FF")

plt.tight_layout()
plt.show()

cosmic_card("""
<p><b>Min-Max</b> приводит значения к диапазону от 0 до 1.</p>
<p><b>Z-score</b> преобразует данные так, чтобы среднее было равно
0, а стандартное отклонение — 1.</p>
<p>Эти преобразования меняют масштаб данных, но не гарантируют
нормальность распределения.</p>
""", title="Вывод по нормализации")

# %% [82]
# @title Проверка нормальности распределений

from scipy import stats

normality_results = []

for name, values in [
    ("Логарифм массы", log_mass),
    ("Логарифм периода", log_period)
]:
    statistic, p_value = stats.normaltest(values)

    normality_results.append({
        "Величина": name,
        "Статистика теста": statistic,
        "p-value": p_value,
        "Результат": (
            "Нет оснований отвергнуть нормальность"
            if p_value >= 0.05
            else "Нормальность отвергается"
        )
    })

normality_df = pd.DataFrame(normality_results)

styled_html_table(
    normality_df,
    "РЕЗУЛЬТАТЫ ТЕСТА НА НОРМАЛЬНОСТЬ",
    digits=6
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

stats.probplot(log_mass, dist="norm", plot=axes[0])
stats.probplot(log_period, dist="norm", plot=axes[1])

axes[0].set_title("Q-Q график логарифма массы", color="#D4C4FF")
axes[1].set_title("Q-Q график логарифма периода", color="#D4C4FF")

for ax, color in zip(axes, [COLORS["mass"], COLORS["period"]]):
    ax.get_lines()[0].set_color(color)
    ax.get_lines()[0].set_markerfacecolor(color)
    ax.get_lines()[0].set_markeredgecolor(color)
    ax.get_lines()[1].set_color(COLORS["pink"])
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

cosmic_card("""
<p>
Тест Д'Агостино проверяет, насколько распределение отличается
от нормального по асимметрии и эксцессу.
</p>
<p>
Если p-value меньше 0,05, нулевая гипотеза о нормальности
отвергается. Если p-value не меньше 0,05, данных недостаточно,
чтобы отвергнуть эту гипотезу. Это не доказывает, что распределение
точно нормальное.
</p>
""", title="Интерпретация проверки нормальности")

# %% [83]
# @title Корреляционный анализ массы и периода

pearson_r, pearson_p = stats.pearsonr(log_mass, log_period)
spearman_r, spearman_p = stats.spearmanr(mass, period)

correlation_results = pd.DataFrame([
    {
        "Метод": "Пирсон",
        "Коэффициент": pearson_r,
        "p-value": pearson_p
    },
    {
        "Метод": "Спирмен",
        "Коэффициент": spearman_r,
        "p-value": spearman_p
    }
])

styled_html_table(
    correlation_results,
    "КОРРЕЛЯЦИОННЫЙ АНАЛИЗ",
    digits=6
)

fig, ax = plt.subplots(figsize=(11, 7))

ax.scatter(
    log_mass,
    log_period,
    s=13,
    alpha=0.28,
    color=COLORS["mass"],
    edgecolors="none"
)

slope, intercept = np.polyfit(log_mass, log_period, 1)
x_line = np.linspace(log_mass.min(), log_mass.max(), 200)

ax.plot(
    x_line,
    slope * x_line + intercept,
    color=COLORS["pink"],
    linewidth=2.5,
    label="Линейный тренд"
)

ax.set_title(
    "Связь логарифма массы и периода обращения",
    color="#D4C4FF",
    fontweight="bold"
)
ax.set_xlabel("log₁₀(массы Юпитера)")
ax.set_ylabel("log₁₀(периода в сутках)")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

cosmic_card(f"""
<p>Коэффициент Пирсона: <b>{pearson_r:.4f}</b>.</p>
<p>Коэффициент Спирмена: <b>{spearman_r:.4f}</b>.</p>
<p>p-value Спирмена: <b>{spearman_p:.6g}</b>.</p>
<p>
Корреляция показывает наличие или отсутствие статистической связи,
но сама по себе не доказывает причинно-следственную зависимость.
</p>
""", title="Вывод по корреляционному анализу")

# %% [84]
# @title Перестановочный тест корреляции Спирмена

SEED = 42
N_PERMUTATIONS = 3000
MAX_SAMPLE = 3000
ALPHA = 0.05

rng = np.random.default_rng(SEED)

sample_size = min(MAX_SAMPLE, len(df))

sample_idx = rng.choice(
    len(df),
    size=sample_size,
    replace=False
)

x = mass.iloc[sample_idx].to_numpy()
y = period.iloc[sample_idx].to_numpy()

observed_corr = stats.spearmanr(x, y).statistic

permuted_corrs = np.empty(N_PERMUTATIONS)

for i in range(N_PERMUTATIONS):
    shuffled_y = rng.permutation(y)
    permuted_corrs[i] = stats.spearmanr(x, shuffled_y).statistic

permutation_p = (
    np.sum(np.abs(permuted_corrs) >= abs(observed_corr)) + 1
) / (N_PERMUTATIONS + 1)

decision = (
    "Нулевая гипотеза отвергается: обнаружена статистическая связь"
    if permutation_p < ALPHA
    else "Нет оснований отвергнуть нулевую гипотезу"
)

hypothesis_result = pd.DataFrame([{
    "Размер подвыборки": sample_size,
    "Количество перестановок": N_PERMUTATIONS,
    "Наблюдаемая корреляция": observed_corr,
    "p-value": permutation_p,
    "Уровень значимости": ALPHA,
    "Решение": decision
}])

styled_html_table(
    hypothesis_result,
    "РЕЗУЛЬТАТ ПЕРЕСТАНОВОЧНОГО ТЕСТА",
    digits=6
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.hist(
    permuted_corrs,
    bins=45,
    color=COLORS["accent"],
    edgecolor="#E8EDFF",
    alpha=0.85
)

ax.axvline(
    observed_corr,
    color=COLORS["pink"],
    linewidth=2.5,
    label=f"Наблюдаемая корреляция: {observed_corr:.3f}"
)

ax.axvline(
    -observed_corr,
    color=COLORS["gold"],
    linewidth=2,
    linestyle="--",
    label="Симметричное значение"
)

ax.set_title(
    "Нулевое распределение перестановочного теста",
    color="#D4C4FF",
    fontweight="bold"
)
ax.set_xlabel("Коэффициент корреляции Спирмена")
ax.set_ylabel("Количество перестановок")
ax.legend(frameon=False)
ax.grid(axis="y")

plt.tight_layout()
plt.show()

cosmic_card(f"""
<p><b>Нулевая гипотеза:</b> между массой и периодом обращения
нет статистической зависимости.</p>
<p><b>Альтернативная гипотеза:</b> статистическая зависимость существует.</p>
<p>Полученное p-value: <b>{permutation_p:.6g}</b>.</p>
<p>{decision}.</p>
<p>
Тест проведён на случайной подвыборке из {sample_size} объектов
с использованием {N_PERMUTATIONS} перестановок.
</p>
""", title="Интерпретация проверки гипотезы")

# %% [85]
# @title Сравнение генеральных и выборочных статистик

population = log_mass.to_numpy()
N = len(population)

population_mean = population.mean()
population_variance = population.var(ddof=0)
population_std = population.std(ddof=0)

population_stats = pd.DataFrame([{
    "Объём генеральной совокупности": N,
    "Среднее": population_mean,
    "Дисперсия": population_variance,
    "Стандартное отклонение": population_std
}])

styled_html_table(
    population_stats,
    "СТАТИСТИКИ ГЕНЕРАЛЬНОЙ СОВОКУПНОСТИ",
    digits=6
)

rng = np.random.default_rng(SEED + 100)

sample_sizes = [
    n for n in [10, 30, 50, 100, 300, 500, 1000]
    if n <= N
]

REPEATS = 500
sampling_rows = []

for mode, replace in [
    ("С возвращением", True),
    ("Без возвращения", False)
]:
    for n in sample_sizes:
        means = []
        variances = []

        for _ in range(REPEATS):
            sample = rng.choice(
                population,
                size=n,
                replace=replace
            )

            means.append(np.mean(sample))
            variances.append(np.var(sample, ddof=1))

        means = np.asarray(means)
        variances = np.asarray(variances)

        sampling_rows.append({
            "Способ отбора": mode,
            "Размер выборки": n,
            "Среднее выборочных средних": means.mean(),
            "Смещение среднего": means.mean() - population_mean,
            "RMSE среднего": np.sqrt(
                np.mean((means - population_mean) ** 2)
            ),
            "Средняя выборочная дисперсия": variances.mean(),
            "Дисперсия генеральной совокупности": population_variance
        })

consistency_df = pd.DataFrame(sampling_rows)

styled_html_table(
    consistency_df,
    "СРАВНЕНИЕ ВЫБОРОЧНЫХ ОЦЕНОК",
    digits=6
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for mode, color in [
    ("С возвращением", COLORS["mass"]),
    ("Без возвращения", COLORS["period"])
]:
    subset = consistency_df[
        consistency_df["Способ отбора"] == mode
    ]

    axes[0].plot(
        subset["Размер выборки"],
        subset["RMSE среднего"],
        marker="o",
        linewidth=2.5,
        color=color,
        label=mode
    )

    axes[1].plot(
        subset["Размер выборки"],
        subset["Смещение среднего"],
        marker="o",
        linewidth=2.5,
        color=color,
        label=mode
    )

axes[0].set_title("RMSE выборочного среднего", color="#D4C4FF")
axes[0].set_xlabel("Размер выборки")
axes[0].set_ylabel("RMSE")
axes[0].legend(frameon=False)
axes[0].grid(alpha=0.3)

axes[1].set_title("Смещение выборочного среднего", color="#D4C4FF")
axes[1].set_xlabel("Размер выборки")
axes[1].set_ylabel("Смещение")
axes[1].axhline(0, color=COLORS["gold"], linestyle="--")
axes[1].legend(frameon=False)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

cosmic_card("""
<p>
При увеличении размера выборки оценки среднего обычно становятся
точнее: их RMSE уменьшается. Это служит эмпирической иллюстрацией
состоятельности выборочного среднего.
</p>
<p>
При отборе без возвращения один объект не может повториться
внутри одной выборки. Поэтому разброс оценок обычно меньше,
чем при отборе с возвращением.
</p>
""", title="Вывод о состоятельности оценок")

# %% [86]
# @title Математическая база распределений

distribution_theory_content = f"""
<div style="
    color:{COLORS['period']};
    font-size:13px;
    letter-spacing:2px;
    font-weight:bold;
    margin-bottom:12px;
">
    STATISTICS · EXOPLANETS
</div>

<p style="
    color:#DDE5FF;
    font-size:16px;
    line-height:1.7;
    margin-bottom:24px;
">
    В исследовании рассматриваются эмпирические распределения массы
    и периода обращения экзопланет, а также их логарифмированные значения.
</p>

<div style="
    padding:20px;
    margin-bottom:16px;
    border-radius:14px;
    background:rgba(145,175,255,.09);
    border:1px solid rgba(145,175,255,.25);
">
    <div style="
        color:{COLORS['mass']};
        font-size:17px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        01 · Математическое ожидание
    </div>

    <p style="color:#DDE5FF;line-height:1.6;">
        Для дискретной случайной величины математическое ожидание определяется как:
    </p>

    <div style="
        color:{COLORS['mass']};
        font-size:23px;
        text-align:center;
        padding:12px 0;
    ">
        E(X) = ∑ xᵢ pᵢ
    </div>

    <p style="color:#BFCBEA;font-size:14px;line-height:1.6;margin-bottom:0;">
        <b>xᵢ</b> — возможные значения величины,
        <b>pᵢ</b> — вероятности этих значений.
    </p>
</div>

<div style="
    padding:20px;
    margin-bottom:16px;
    border-radius:14px;
    background:rgba(201,182,255,.09);
    border:1px solid rgba(201,182,255,.25);
">
    <div style="
        color:{COLORS['accent']};
        font-size:17px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        02 · Дисперсия
    </div>

    <p style="color:#DDE5FF;line-height:1.6;">
        Дисперсия характеризует разброс значений относительно математического ожидания:
    </p>

    <div style="
        color:{COLORS['accent']};
        font-size:23px;
        text-align:center;
        padding:12px 0;
    ">
        D(X) = E[(X − E(X))²]
    </div>
</div>

<div style="
    padding:20px;
    margin-bottom:16px;
    border-radius:14px;
    background:rgba(114,229,192,.08);
    border:1px solid rgba(114,229,192,.25);
">
    <div style="
        color:{COLORS['period']};
        font-size:17px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        03 · Непрерывные случайные величины
    </div>

    <p style="color:#DDE5FF;line-height:1.7;margin-bottom:0;">
        Для непрерывной случайной величины вероятности задаются плотностью
        распределения <b>f(x)</b>, а математическое ожидание и дисперсия
        определяются с помощью интегралов.
    </p>
</div>

<div style="
    padding:20px;
    margin-bottom:16px;
    border-radius:14px;
    background:rgba(255,209,138,.08);
    border:1px solid rgba(255,209,138,.25);
">
    <div style="
        color:{COLORS['gold']};
        font-size:17px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        04 · Логарифмирование данных
    </div>

    <p style="color:#DDE5FF;line-height:1.7;">
        Масса и период обращения экзопланет являются положительными величинами.
        В каталоге они могут иметь сильно различающиеся масштабы, поэтому
        для анализа дополнительно используются десятичные логарифмы.
    </p>

    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:10px;
        margin:16px 0;
    ">
        <div style="
            flex:1;
            min-width:180px;
            padding:13px;
            border-radius:10px;
            background:rgba(145,175,255,.12);
            color:{COLORS['mass']};
            text-align:center;
        ">
            log₁₀(масса)
        </div>
        <div style="
            flex:1;
            min-width:180px;
            padding:13px;
            border-radius:10px;
            background:rgba(114,229,192,.12);
            color:{COLORS['period']};
            text-align:center;
        ">
            log₁₀(период)
        </div>
    </div>

    <p style="color:#DDE5FF;line-height:1.7;margin-bottom:0;">
        Логарифмирование помогает изучать порядок величин и уменьшает влияние
        очень больших значений на визуальное представление данных.
    </p>
</div>

<div style="
    padding:20px;
    border-radius:14px;
    background:rgba(255,145,200,.08);
    border:1px solid rgba(255,145,200,.28);
">
    <div style="
        color:{COLORS['pink']};
        font-size:17px;
        font-weight:bold;
        margin-bottom:12px;
    ">
        05 · Проверка формы распределения
    </div>

    <p style="color:#DDE5FF;line-height:1.7;margin-bottom:12px;">
        Логарифмирование не гарантирует нормальное распределение.
        Поэтому форму логарифмированных данных необходимо проверять отдельно:
    </p>

    <ul style="
        color:#F4F5FF;
        line-height:1.9;
        padding-left:22px;
        margin-bottom:0;
    ">
        <li>гистограммы распределений;</li>
        <li>Q-Q графики;</li>
        <li>тест нормальности.</li>
    </ul>
</div>
"""

cosmic_card(
    distribution_theory_content,
    title="Математическая база распределений"
)

# %% [87]
# @title Итоговые выводы исследования экзопланет

required_vars = [
    "df",
    "mass",
    "period",
    "log_mass",
    "log_period",
    "normality_df",
    "spearman_r",
    "spearman_p",
    "permutation_p",
    "consistency_df"
]

missing_vars = [
    name for name in required_vars
    if name not in globals()
]

if missing_vars:
    raise NameError(
        "Не найдены переменные из предыдущих ячеек: "
        + ", ".join(missing_vars)
        + ". Запусти все ячейки по порядку."
    )

mass_mean = mass.mean()
mass_median = mass.median()
period_mean = period.mean()
period_median = period.median()

normality_text = []

for _, row in normality_df.iterrows():
    normality_text.append(
        f"<li>{html_lib.escape(str(row['Величина']))}: "
        f"{html_lib.escape(str(row['Результат']))} "
        f"(p-value = {row['p-value']:.4g})</li>"
    )

sampling_rmse = (
    consistency_df
    .groupby("Способ отбора")["RMSE среднего"]
    .last()
)

sampling_text = "".join(
    f"<li>{html_lib.escape(str(method))}: "
    f"RMSE = {value:.6f}</li>"
    for method, value in sampling_rmse.items()
)

if permutation_p < 0.05:
    hypothesis_text = (
        "Перестановочный тест выявил статистически значимую "
        "связь между массой и периодом обращения."
    )
else:
    hypothesis_text = (
        "Перестановочный тест не выявил статистически значимой "
        "связи между массой и периодом обращения."
    )

conclusion_content = f"""
<h3 style="color:{COLORS['accent']};">Исследуемые данные</h3>

<p>
В анализ вошло <b>{len(df)}</b> экзопланет из NASA Exoplanet Archive.
Рассматривались наилучшая доступная оценка массы планеты и период
её обращения вокруг звезды.
</p>

<h3 style="color:{COLORS['accent']};">Описательная статистика</h3>

<ul>
<li>Средняя масса: <b>{mass_mean:.6f}</b> масс Юпитера.</li>
<li>Медианная масса: <b>{mass_median:.6f}</b> масс Юпитера.</li>
<li>Средний период: <b>{period_mean:.6f}</b> суток.</li>
<li>Медианный период: <b>{period_median:.6f}</b> суток.</li>
</ul>

<h3 style="color:{COLORS['accent']};">Нормальность распределений</h3>
<ul>{''.join(normality_text)}</ul>

<h3 style="color:{COLORS['accent']};">Зависимость величин</h3>

<p>
Коэффициент Спирмена: <b>{spearman_r:.5f}</b>.
p-value: <b>{spearman_p:.6g}</b>.
</p>

<p>{hypothesis_text}</p>

<p>
В перестановочном тесте получено p-value =
<b>{permutation_p:.6g}</b>.
</p>

<h3 style="color:{COLORS['accent']};">Состоятельность оценок</h3>
<ul>{sampling_text}</ul>

<h3 style="color:{COLORS['accent']};">Общий вывод</h3>

<p>
В ходе исследования были загружены и обработаны данные об
экзопланетах, изучены распределения массы и периода обращения,
проведены нормализация, проверка нормальности и корреляционный анализ.
</p>

<p>
Также была проверена статистическая связь между величинами
с помощью перестановочного теста и исследовано поведение
выборочного среднего при разных размерах выборки и двух способах
отбора объектов.
</p>

<p>
Полученные результаты описывают только исследуемый каталог
обнаруженных экзопланет и не могут автоматически переноситься
на все планеты Вселенной.
</p>
"""

cosmic_card(
    conclusion_content,
    title="ИТОГИ СТАТИСТИЧЕСКОГО ИССЛЕДОВАНИЯ ЭКЗОПЛАНЕТ"
)

# %% [88]

# @title Дополнительное задание КТвД — интерактивный генератор в Marimo

cosmic_card("""
<style>
.generator-intro {
    color:#DCE6FF;
    font-size:15px;
    line-height:1.8;
    margin-bottom:22px;
}
.generator-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:12px;
    margin:20px 0 28px;
}
.generator-item {
    padding:19px;
    border-radius:14px;
    background:linear-gradient(145deg,rgba(125,100,230,.17),rgba(65,150,230,.07));
    border:1px solid rgba(180,195,255,.25);
    border-top:3px solid var(--accent);
}
.generator-item h3 {
    color:var(--accent);
    font-size:16px;
    margin:0 0 11px;
}
.generator-item p {
    margin:0;
    color:#E5E9FA;
    font-size:14px;
    line-height:1.7;
}
.generator-heading {
    display:flex;
    align-items:center;
    gap:12px;
    margin:30px 0 17px;
}
.generator-heading h2 {
    color:#C9B6FF;
    font-size:23px;
    margin:0;
}
.generator-info {
    margin:13px 0;
    padding:19px 22px;
    border-radius:13px;
    background:rgba(255,255,255,.045);
    border:1px solid rgba(180,195,255,.18);
    border-left:3px solid var(--accent);
    color:#F4F5FF;
    line-height:1.8;
}
.generator-info p {
    margin:0;
}
.generator-note {
    color:#AFC4E8;
    font-size:12px;
    letter-spacing:1.5px;
    margin:0 0 12px;
    font-weight:bold;
}
.generator-info code {
    color:#C9B6FF;
}
.generator-info pre {
    padding:13px;
    border-radius:9px;
    background:rgba(0,0,0,.25);
    color:#B9D9FF;
    overflow-x:auto;
    margin-top:12px;
}
.generator-result {
    padding:20px 23px;
    border-radius:14px;
    background:linear-gradient(120deg,rgba(126,224,195,.12),rgba(122,162,247,.09));
    border:1px solid rgba(126,224,195,.3);
    color:#E8FFF7;
    line-height:1.8;
    margin-top:20px;
}
.generator-link {
    color:#B9D9FF;
    text-decoration:underline;
    overflow-wrap:anywhere;
}
</style>

<h2 style="color:#C9B6FF;font-size:27px;margin:0 0 12px;">
    Интерактивный генератор случайных выборок
</h2>

<p class="generator-intro">
    В дополнительной части работы был разработан интерактивный генератор
    случайных выборок на основе датасета NASA Exoplanet Archive.
    Реализация выполнена на Python в среде <code style="color:#B9D9FF;">Marimo</code>.
    Генератор позволяет получать случайные выборки из исходных наблюдений
    и исследовать, насколько их статистические характеристики
    соответствуют характеристикам исходных данных.
</p>

<div class="generator-grid">
    <div class="generator-item" style="--accent:#C9B6FF;">
        <h3>Масса экзопланеты</h3>
        <p>
            Масса планеты, выраженная в массах Юпитера.
            Для неё реализована отдельная процедура генерации случайной выборки.
        </p>
    </div>

    <div class="generator-item" style="--accent:#7AA2F7;">
        <h3>Период обращения</h3>
        <p>
            Продолжительность обращения планеты вокруг своей звезды,
            выраженная в земных сутках. Для этой величины также реализован генератор.
        </p>
    </div>

    <div class="generator-item" style="--accent:#7EE0C3;">
        <h3>Интерактивное управление</h3>
        <p>
            Пользователь может выбирать количество генерируемых значений
            и задавать Seed для воспроизведения эксперимента.
        </p>
    </div>
</div>

<div style="height:1px;background:linear-gradient(90deg,transparent,#8B7BCB,transparent);margin:30px 0;"></div>

<div class="generator-heading">
    <h2>Принцип работы генератора</h2>
</div>

<div class="generator-info" style="--accent:#7AA2F7;">
    <div class="generator-note">ЭМПИРИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ</div>
    <p>
        В качестве метода генерации используется выборка с возвращением.
        Исходные значения упорядочиваются, после чего генератор случайным
        образом выбирает индексы наблюдений. Выбранные значения не удаляются
        из исходной выборки, поэтому одно и то же наблюдение может встретиться
        несколько раз. Таким образом, генерация воспроизводит эмпирическое
        распределение исходных данных.
    </p>
</div>

<div class="generator-info" style="--accent:#C9B6FF;">
    <div class="generator-note">УПРАВЛЕНИЕ СЛУЧАЙНОСТЬЮ</div>
    <p>
        Для получения случайных индексов используется генератор
        <code>numpy.random.default_rng()</code>.
        Параметр Seed задаёт начальное состояние генератора:
        при одинаковом Seed и одинаковом количестве генерируемых значений
        последовательность результатов воспроизводится.
    </p>
</div>

<div class="generator-info" style="--accent:#7EE0C3;">
    <div class="generator-note">МАТЕМАТИЧЕСКАЯ ОСНОВА</div>
    <p>
        Каждое исходное наблюдение имеет одинаковую вероятность выбора
        1/N, где N — количество наблюдений в исходной выборке.
        При увеличении размера генерируемой выборки её статистические
        характеристики в среднем приближаются к характеристикам
        исходного эмпирического распределения.
    </p>
</div>

<div class="generator-heading">
    <h2>Проверка корректности</h2>
</div>

<div class="generator-grid">
    <div class="generator-item" style="--accent:#7EE0C3;">
        <h3>Статистические метрики</h3>
        <p>
            Сравниваются среднее, медиана, дисперсия, стандартное отклонение,
            минимумы, максимумы и квантили исходных и сгенерированных данных.
        </p>
    </div>

    <div class="generator-item" style="--accent:#7AA2F7;">
        <h3>Распределения</h3>
        <p>
            Строятся гистограммы и графики для визуального сопоставления
            исходных и сгенерированных значений.
        </p>
    </div>

    <div class="generator-item" style="--accent:#C9B6FF;">
        <h3>Автоматические тесты</h3>
        <p>
            Проверяются количество значений, принадлежность исходной выборке,
            воспроизводимость Seed и близость основных статистических характеристик.
        </p>
    </div>
</div>

<div class="generator-info" style="--accent:#F7768E;">
    <div class="generator-note">ОГРАНИЧЕНИЯ МЕТОДА</div>
    <p>
        Генератор воспроизводит эмпирическое распределение, поэтому может
        выдавать только значения, присутствующие в исходном датасете.
        Он не создаёт новые значения за пределами наблюдаемой выборки
        и наследует возможные ограничения исходных данных.
    </p>
</div>

<div style="height:1px;background:linear-gradient(90deg,transparent,#8B7BCB,transparent);margin:30px 0;"></div>

<div class="generator-heading">
    <h2>Как открыть и запустить генератор</h2>
</div>

<div class="generator-info" style="--accent:#C9B6FF;">
    <div class="generator-note">ШАГ 1. СКАЧАТЬ ФАЙЛЫ</div>
    <p>
        Перейдите в репозиторий GitHub. Для запуска понадобятся два файла:
        <code>notebook.py</code> — исходный код генератора;
        <code>exoplanets_mass_period.csv</code> — исходный датасет.
        Скачайте оба файла и поместите их в одну папку.
    </p>
    <p style="margin-top:12px;">
        <a class="generator-link"
           href="https://github.com/sofia28s/statistics/blob/main/notebook.py"
           target="_blank">Открыть notebook.py</a>
        <br>
        <a class="generator-link"
           href="https://github.com/sofia28s/statistics/blob/main/exoplanets_mass_period.csv"
           target="_blank">Открыть исходный CSV</a>
    </p>
</div>

<div class="generator-info" style="--accent:#7AA2F7;">
    <div class="generator-note">ШАГ 2. УСТАНОВИТЬ НЕОБХОДИМЫЕ БИБЛИОТЕКИ</div>
    <p>Откройте терминал и выполните команду:</p>
    <pre>pip install marimo numpy pandas matplotlib</pre>
</div>

<div class="generator-info" style="--accent:#7EE0C3;">
    <div class="generator-note">ШАГ 3. ЗАПУСТИТЬ MARIMO</div>
    <p>
        Откройте терминал в папке, где находятся скачанные файлы,
        и выполните команду:
    </p>
    <pre>marimo edit notebook.py</pre>
    <p>
        После запуска откроется интерактивный ноутбук Marimo в браузере.
    </p>
</div>

<div class="generator-info" style="--accent:#C9B6FF;">
    <div class="generator-note">ШАГ 4. ЗАГРУЗИТЬ ДАННЫЕ И ПРОВЕСТИ ГЕНЕРАЦИЮ</div>
    <p>
        В открывшемся ноутбуке загрузите файл
        <code>exoplanets_mass_period.csv</code> через интерактивный элемент
        загрузки CSV. Затем выберите количество генерируемых значений
        и Seed с помощью ползунков. Результаты, таблицы, графики
        и автоматические проверки будут обновляться при изменении параметров.
    </p>
</div>

<div class="generator-result">
    <div class="generator-note" style="color:#7EE0C3;">РЕПОЗИТОРИЙ И ФАЙЛЫ ПРОЕКТА</div>
    <p>
        <b>Исходный код:</b> notebook.py<br>
        <b>Датасет:</b> exoplanets_mass_period.csv
    </p>
    <p style="margin-top:12px;">
        <b>GitHub:</b>
        <a class="generator-link"
           href="https://github.com/sofia28s/statistics"
           target="_blank">
            Открыть репозиторий с генератором
        </a>
    </p>
</div>

<div class="generator-result">
    <div class="generator-note" style="color:#7EE0C3;">ИТОГ</div>
    <p>
        В результате создан интерактивный генератор случайных выборок
        для двух характеристик экзопланет — массы и периода обращения.
        Сравнение статистических характеристик, визуализация распределений
        и автоматические тесты позволяют оценить корректность генерации
        и способность метода воспроизводить эмпирическое распределение
        исходных данных.
    </p>
</div>
""")


# %% [89]

# @title Соответствие критериям оценивания КТвД

criteria_content = f"""
<div style="
    color:#72E5C2;
    font-size:14px;
    letter-spacing:2px;
    font-weight:bold;
    margin-bottom:12px;
">
    КТВД · ДОПОЛНИТЕЛЬНОЕ ИССЛЕДОВАНИЕ
</div>

<p style="color:#DDE5FF;font-size:16px;line-height:1.7;margin-bottom:24px;">
    Ниже представлено соответствие выполненной работы критериям
    дополнительного задания по курсу КТвД. Помимо статистического
    исследования, реализован интерактивный генератор случайных выборок
    в Marimo.
</p>

<table style="
    width:100%;
    border-collapse:collapse;
    font-size:14px;
    line-height:1.6;
">
    <thead>
        <tr>
            <th style="padding:13px;text-align:left;color:#C9B6FF;
                       border-bottom:1px solid rgba(201,182,255,.45);">
                Критерий
            </th>
            <th style="padding:13px;text-align:left;color:#C9B6FF;
                       border-bottom:1px solid rgba(201,182,255,.45);">
                Где выполнено
            </th>
        </tr>
    </thead>
    <tbody>
        {''.join(
            f'''
            <tr>
                <td style="padding:12px 13px;color:#F4F5FF;
                           border-bottom:1px solid rgba(180,195,255,.16);">
                    {criterion}
                </td>
                <td style="padding:12px 13px;color:#72E5C2;
                           border-bottom:1px solid rgba(180,195,255,.16);">
                    {location}
                </td>
            </tr>
            '''
            for criterion, location in [
                ("Отдельный датасет по выбранной теме",
                 "Загрузка данных NASA Exoplanet Archive и EDA"),
                ("Две случайные величины",
                 "Математическая база распределений"),
                ("Обоснование случайной природы величин",
                 "Математическая база распределений"),
                ("Разведочный анализ данных (EDA)",
                 "Загрузка данных NASA Exoplanet Archive и EDA"),
                ("Нормализация и стандартизация",
                 "Нормализация и стандартизация данных"),
                ("Исследование распределений",
                 "Математическая база распределений; Проверка нормальности распределений"),
                ("Проверка статистической зависимости",
                 "Корреляционный анализ массы и периода"),
                ("Проверка гипотезы",
                 "Перестановочный тест корреляции Спирмена"),
                ("Сравнение генеральной и выборочных статистик",
                 "Сравнение генеральных и выборочных статистик"),
                ("Выборки с возвращением и без возвращения",
                 "Сравнение генеральных и выборочных статистик"),
                ("Эмпирическая проверка состоятельности оценок",
                 "Сравнение генеральных и выборочных статистик"),
                ("Реализация генератора случайных выборок",
                 "Отдельный интерактивный генератор в Marimo (GitHub)"),
                ("Генерация выборок для двух случайных величин",
                 "Генератор значений массы и орбитального периода экзопланет"),
                ("Настройка размера выборки и начального значения Seed",
                 "Интерактивные элементы управления в Marimo"),
                ("Проверка воспроизводимости генерации",
                 "Автоматические тесты генератора в Marimo"),
                ("Сравнение статистик исходных и сгенерированных данных",
                 "Таблицы, графики и оценка относительных отклонений в Marimo"),
                ("Проверка сходимости статистических характеристик",
                 "График изменения среднего значения при увеличении размера выборки в Marimo"),
                ("Итоговая интерпретация результатов",
                 "Итоговые выводы исследования экзопланет")
            ]
        )}
    </tbody>
</table>
"""

cosmic_card(
    criteria_content,
    title="Соответствие критериям оценивания дополнительного задания КТвД"
)


# %% [90]
#@title
from IPython.display import HTML

BACKGROUND_URL = (
    "https://raw.githubusercontent.com/sofia28s/statistics/"
    "refs/heads/main/background.png"
)
THANKS_URL = (
    "https://raw.githubusercontent.com/sofia28s/statistics/"
    "main/thanks.jpg"
)

display(HTML(f"""
<div style="
    background: linear-gradient(rgba(5, 10, 30, 0.25), rgba(5, 10, 30, 0.25)),
        url('{BACKGROUND_URL}');
    background-size: cover;
    background-position: center;
    padding: 35px;
    border-radius: 18px;
    text-align: center;
">
    <img src="{THANKS_URL}"
         style="width: 300px; max-width: 100%; height: auto; border-radius: 12px;">
</div>
"""))

