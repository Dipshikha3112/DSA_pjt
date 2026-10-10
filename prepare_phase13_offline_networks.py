"""Create cached OSMnx-compatible GraphML networks from regional Geofabrik PBFs.

Offline, no Overpass. Extracts ways intersecting the radius with a small buffer,
then clips graph by node coordinates and retains largest strongly connected component.
Records provenance and approximations for reproducibility.
"""
import argparse
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import networkx as nx
import osmnx as ox
import osmium

AREAS = {
    'chennai': ('southern-zone-latest.osm.pbf', 13.0067, 80.2570, 'Adyar, Chennai'),
    'jaipur': ('northern-zone-latest.osm.pbf', 26.9167, 75.7878, 'Civil Lines, Jaipur'),
    'kolkata': ('eastern-zone-latest.osm.pbf', 22.5687, 88.4335, 'Salt Lake Sector V, Kolkata'),
}
# These centers are explicit reproducible approximations, NOT automatically geocoded OSMnx addresses.
ALLOWED = {'motorway','trunk','primary','secondary','tertiary','unclassified','residential','living_street',
           'motorway_link','trunk_link','primary_link','secondary_link','tertiary_link','service'}
SPEED = {'motorway':80,'trunk':60,'primary':50,'secondary':40,'tertiary':35,'unclassified':30,
         'residential':25,'living_street':10,'service':15,'motorway_link':40,'trunk_link':30,
         'primary_link':25,'secondary_link':25,'tertiary_link':20}

def meters(lat1,lon1,lat2,lon2):
    r=6371008.8
    p1,p2=math.radians(lat1),math.radians(lat2)
    dp=p2-p1;dl=math.radians(lon2-lon1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1,math.sqrt(a)))

def road_class(tags):
    value=tags.get('highway','')
    if value in ALLOWED:return value
    return None

def speed_kph(tags, highway):
    raw=tags.get('maxspeed','')
    try:
        if raw and raw.split()[0].isdigit():
            v=float(raw.split()[0]);return v*1.609344 if 'mph' in raw.lower() else v
    except (ValueError,IndexError):pass
    return SPEED.get(highway,25)

class Collector(osmium.SimpleHandler):
    def __init__(self, lat,lon,radius):
        super().__init__();self.lat=lat;self.lon=lon;self.radius=radius;self.g=nx.MultiDiGraph()
        self.seen=0;self.accepted=0
    def way(self,w):
        tags={t.k:t.v for t in w.tags}
        hw=road_class(tags)
        if hw is None or tags.get('area')=='yes':return
        if tags.get('access') in ('private','no') or tags.get('motor_vehicle') in ('private','no') or tags.get('motorcar') in ('private','no'):return
        try:
            coords=[(int(n.ref),float(n.lat),float(n.lon)) for n in w.nodes if n.location.valid()]
        except (ValueError,RuntimeError):return
        if len(coords)<2:return
        self.seen+=1
        oneway=tags.get('oneway','').lower()
        reverse=oneway in ('-1','reverse')
        forward=oneway in ('yes','true','1') or hw in ('motorway','motorway_link') or tags.get('junction')=='roundabout'
        bidirectional=not (forward or reverse)
        sp=speed_kph(tags,hw)
        if sp<=0:return
        for a,b in zip(coords[:-1],coords[1:]):
            u,lat1,lon1=a;v,lat2,lon2=b
            if min(meters(self.lat,self.lon,lat1,lon1),meters(self.lat,self.lon,lat2,lon2))>self.radius:continue
            length=meters(lat1,lon1,lat2,lon2)
            if length<0.01:continue
            for node,lat,lon in (a,b):
                self.g.add_node(node,x=lon,y=lat)
            attrs=dict(osmid=str(w.id),length=float(length),highway=hw,speed_kph=float(sp),travel_time=float(length*3.6/sp))
            if reverse:self.g.add_edge(v,u,**attrs)
            else:self.g.add_edge(u,v,**attrs)
            if bidirectional:self.g.add_edge(v,u,**attrs)
            self.accepted+=1

def run_one(area, args):
    filename,lat,lon,place=AREAS[area]
    pbf=Path(args.extracts)/filename
    if not pbf.is_file():raise FileNotFoundError(f'Missing extract: {pbf}')
    target=Path(args.cache)/f'phase13_{area}.graphml'
    if target.exists() and not args.overwrite:
        print(f'{area}: existing graph preserved: {target}',flush=True);return
    print(f'{area}: scanning {pbf.name} ({pbf.stat().st_size/1e6:.0f} MB)',flush=True)
    handler=Collector(lat,lon,args.radius)
    handler.apply_file(str(pbf),locations=True,idx='flex_mem')
    g=handler.g
    if g.number_of_nodes()==0:raise RuntimeError(f'{area}: no roads extracted')
    # OSMnx graph_from_address(dist=1200) extracts a bbox then truncates;
    # this offline extraction uses a radial inclusion criterion, so it is not identical.
    keep={n for n,d in g.nodes(data=True) if meters(lat,lon,d['y'],d['x'])<=args.radius}
    g=g.subgraph(keep).copy()
    g.remove_nodes_from(list(nx.isolates(g)))
    if not g.number_of_nodes():raise RuntimeError(f'{area}: no roads inside radius')
    # Keep largest strongly connected component to ensure OD reachability.
    comps=list(nx.strongly_connected_components(g))
    largest=max(comps,key=len)
    g=g.subgraph(largest).copy()
    if g.number_of_nodes()<10:raise RuntimeError(f'{area}: road graph unexpectedly small: {g.number_of_nodes()} nodes')
    g.graph.update(crs='epsg:4326',simplified=False,source='Geofabrik regional PBF / PyOsmium',area=area)
    target.parent.mkdir(parents=True,exist_ok=True)
    ox.io.save_graphml(g,filepath=target)
    check=ox.io.load_graphml(target)
    if not all('travel_time' in d for *_,d in check.edges(keys=True,data=True)):
        raise RuntimeError(f'{area}: GraphML validation failed')
    meta=dict(area=area,place=place,center_lat=lat,center_lon=lon,radius_m=args.radius,
              input_file=str(pbf),input_bytes=pbf.stat().st_size,source='Geofabrik OSM PBF',
              nodes=g.number_of_nodes(),directed_edges=g.number_of_edges(),
              extracted_segments=handler.accepted,generated_at_utc=datetime.now(timezone.utc).isoformat(),
              assumptions=['Approximate explicitly specified coordinates, not OSMnx geocoding',
                           'Straight-line node-to-node edge lengths, not full OSMnx geometries',
                           'Heuristic speed limits, with maxspeed used when numeric',
                           'Radial clipping and largest strongly connected component',
                           'Simplified access/oneway handling; turn restrictions not modeled',
                           'Network differs methodologically from OSMnx graph_from_address'])
    (target.parent/f'phase13_{area}_offline_provenance.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(f'{area}: SAVED {target}: {g.number_of_nodes()} nodes, {g.number_of_edges()} directed edges',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--extracts',default='data/osm_extracts')
    p.add_argument('--cache',default='data/real_networks')
    p.add_argument('--areas',default='chennai,jaipur,kolkata')
    p.add_argument('--radius',type=int,default=1200)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    for area in args.areas.split(','):
        if area not in AREAS:raise ValueError(f'Unknown area {area}')
        run_one(area,args)
    print('Offline GraphML preparation complete.',flush=True)
if __name__=='__main__':main()
