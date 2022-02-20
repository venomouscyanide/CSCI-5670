"""
Course : CSCI 5760
Author : Paul Louis <paul.louis@ontariotechu.net>
"""
import random
import time
from queue import SimpleQueue
from typing import Set
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import networkx as nx
from bs4 import BeautifulSoup
from networkx import write_gexf

MAX_URLs = 100000


def _print_log(url, seen):
    print(f"Processing node: {url} and at total number of urls: {len(seen)}")


def _queue_helper(focus_node: str, neigh_node: str, queue: SimpleQueue, seen: Set[str], graph: nx.DiGraph):
    graph.add_edge(focus_node, neigh_node)
    queue.put(neigh_node)
    seen.add(neigh_node)


def crawl_away(root: str):
    graph = nx.DiGraph()
    queue = SimpleQueue()
    seen = set()

    queue.put(root)
    bfs_seq = []

    seen.add(root)
    _print_log(root, seen)

    while not queue.empty() and len(seen) < MAX_URLs and len(bfs_seq) < 500:
        focus_node = queue.get()
        bfs_seq.append(focus_node)

        sleep_time = random.randint(0, 3)
        print(f'random sleep of {sleep_time} seconds')
        time.sleep(sleep_time)

        try:
            with urlopen(focus_node) as website_html:
                soup = BeautifulSoup(website_html, 'html.parser')

                for link in soup.find_all('a'):
                    neigh_node = link.get('href')

                    if not neigh_node:
                        continue

                    if neigh_node.lower().endswith('.pdf') or neigh_node.lower().endswith('.jpeg') or \
                            neigh_node.lower().endswith('.jpg'):
                        # need to find more ext to avoid
                        continue

                    if neigh_node.startswith('/'):
                        # print(f"Parent: {focus_node}, neigh_node: {neigh_node},
                        # Joined: {urljoin(focus_node, neigh_node)}")
                        neigh_node = urljoin(focus_node, neigh_node)

                    if neigh_node in seen:
                        continue

                    parsed_url = urlparse(neigh_node)
                    netloc = parsed_url.netloc

                    if 'http' in parsed_url[0] and ('ontariotechu' in netloc or 'uoit' in netloc):
                        _print_log(neigh_node, seen)
                        _queue_helper(focus_node, neigh_node, queue, seen, graph)

                    if len(seen) % 5000 == 0:
                        write_gexf(graph, f'interim_{len(seen)}_len_graph.gexf')
        except Exception as e:
            print(f"Failed: {focus_node} with exception: {e}")

    write_gexf(graph, f'final_{len(seen)}_len_graph.gexf')

    for website in bfs_seq:
        print(website)
    return graph


if __name__ == '__main__':
    root = 'https://ontariotechu.ca/'
    graph = crawl_away(root)
