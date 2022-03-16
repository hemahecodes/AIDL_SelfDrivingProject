import pandas as pd
import argparse as ap
import plotly.express as px
import plotly.graph_objects as go

def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-j',
                        '--json_file', required=True)
    args = parser.parse_args()
    return args

def analyze_db(path_json):
    annotations = pd.read_json(path_json)
    weather = {}
    timeofday={}
    scene = {}
    cat_dict={}
    for attribute in annotations['attributes'].values:
        for key, value in attribute.items():
            if key == 'weather':
                if value in weather:
                    weather[value] = weather[value]+1
                else:
                    weather[value] = 1
            elif key == 'timeofday':
                if value in timeofday:
                    timeofday[value] = timeofday[value] + 1
                else:
                    timeofday[value] = 1
            elif key =='scene':
                if value in scene:
                    scene[value] = scene[value] + 1
                else:
                    scene[value] = 1
    for image in annotations['labels']:
        for object in image:
            if object['category'] not in cat_dict:
                cat_dict[object['category']] = 1
            else:
                cat_dict[object['category']] = cat_dict[object['category']] + 1

    color_discrete_sequence = ['#003f5c','#374c80','#7a5195','#bc5090','#ef5675','#ff764a','#ffa600']
    color_discrete_sequence_categories = ['#4C5454', '#FF715B', '#FFFFFF', '#1EA896','#523F38', '#00241B', '#F7996E', '#E9E3B4', '#95B2B8','#736372','#1A1423', '#FFA69E','#345830']

    cat_dict = {key: value for key, value in sorted(cat_dict.items())}
    weather = {key: value for key, value in sorted(weather.items())}
    timeofday = {key: value for key, value in sorted(timeofday.items())}
    scene = {key: value for key, value in sorted(scene.items())}

    categories_fig = go.Figure()
    categories_fig.add_trace(go.Bar(x=list(cat_dict.keys()), y=list(cat_dict.values()),
                                 marker=dict(color=color_discrete_sequence_categories[0:len(cat_dict) - 1])))
    categories_fig.update_layout(title_text="Number of Instances in each Category")
    categories_fig.write_image('categories_plot.png')

    weather_fig = go.Figure()
    weather_fig.add_trace(go.Bar(x=list(weather.keys()), y=list(weather.values()),marker=dict(color=color_discrete_sequence[0:len(weather)-1])))
    weather_fig.update_layout(title_text="Number of Instances in each Weather")
    weather_fig.write_image('weather_plot.png')

    timeofday_fig = go.Figure()
    timeofday_fig.add_trace(go.Bar(x=list(timeofday.keys()), y=list(timeofday.values()),marker=dict(color=color_discrete_sequence[0:len(timeofday)-1])))
    timeofday_fig.update_layout(title_text="Number of Instances in each Time of Day")
    timeofday_fig.write_image('timeofday_plot.png')

    scene_fig = go.Figure()
    scene_fig.add_trace(go.Bar(x=list(scene.keys()), y=list(scene.values()),marker=dict(color=color_discrete_sequence[0:len(scene)-1])))
    scene_fig.update_layout(title_text="Number of Instances in each Scene")
    scene_fig.write_image('scene_plot.png')


if __name__ == '__main__':
    args = get_args()
    path_json = args.json_file
    analyze_db(path_json)