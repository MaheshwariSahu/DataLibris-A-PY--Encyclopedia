# app.py
from flask import Flask, render_template, request
import pandas as pd
import markdown2
import pickle
from functools import lru_cache
from collections import defaultdict

app = Flask(__name__)

# Load data and model
data = pd.read_csv('encyclopedia.csv')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

search_counts = defaultdict(int)

@lru_cache(maxsize=128)
def get_package_info(package_name):
    import requests
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        info = response.json()['info']
        description = info['description'] or info['summary']
        html_description = markdown2.markdown(description)
        category = model.predict([description])[0]
        return {
            'name': info['name'],
            'version': info['version'],
            'summary': f"Category: {category}",
            'description': html_description,
            'homepage': info['home_page'],
            'predicted_category': category,
            'source': 'pypi'
        }
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/library', methods=['GET', 'POST'])
def library_search():
    info = None
    if request.method == 'POST':
        query = request.form['query'].strip()
        search_counts[query.lower()] += 1
        local = data[data['title'].str.lower() == query.lower()]
        if not local.empty:
            row = local.iloc[0]
            content_html = markdown2.markdown(row['content'])
            info = {
                'name': row['title'],
                'version': 'N/A',
                'summary': f"Category: {row['category']}",
                'description': content_html,
                'homepage': '#',
                'predicted_category': row['category'],
                'source': 'local'
            }
        else:
            info = get_package_info(query)
    return render_template('library.html', info=info)

@app.route('/category', methods=['GET', 'POST'])
def category_search():
    results = []
    page = int(request.args.get('page', 1))
    per_page = 5

    if request.method == 'POST':
        category = request.form['category'].strip().lower()
        matched = data[data['category'].str.lower() == category]
        results = matched.to_dict(orient='records')
        for item in results:
            item['content'] = markdown2.markdown(item['content'])

        start = (page - 1) * per_page
        end = start + per_page
        paginated = results[start:end]
        total_pages = (len(results) + per_page - 1) // per_page

        return render_template('category.html', results=paginated, category=category, page=page, total_pages=total_pages)

    return render_template('category.html', results=[], page=1, total_pages=1)


@app.route('/analytics')
def analytics():
    import plotly.graph_objs as go
    from plotly.offline import plot

    # If no data, fallback to example dataset
    fallback_data = {
        'Pandas': 5,
        'Numpy': 4,
        'Matplotlib': 3,
        'Flask': 2,
        'TensorFlow': 1
    }

    if not search_counts:
        top = sorted(fallback_data.items(), key=lambda x: x[1], reverse=True)
        is_fallback = True
    else:
        top = sorted(search_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        is_fallback = False

    labels, values = zip(*top)
    fig = go.Figure([go.Bar(x=labels, y=values)])
    fig.update_layout(
        title="Top 10 Most Searched Libraries",
        xaxis_title="Library",
        yaxis_title="Search Count"
    )
    graph_div = plot(fig, output_type='div', include_plotlyjs=True)

    return render_template("analytics.html", graph_div=graph_div, is_fallback=is_fallback)



if __name__ == '__main__':
    app.run(debug=True)
