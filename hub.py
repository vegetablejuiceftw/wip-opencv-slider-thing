from sentinelhub import CRS, BBox, WmsRequest, DataSource

from utils import zoom_bbox


# The test site near Võrtsjärv
COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT = (
    (57.9412725, 25.9080495),
    (57.8986439, 26.0045503),
)
print(zoom_bbox(COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT, 1.0))
print((COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT))

# Rivo's instance
RIVO_INSTANCE_ID = '4cb42bbe-04a4-40ad-9c5b-a9a88ba92dc3'
# Fred's instance
INSTANCE_ID = 'c087deea-ec78-4519-9827-230b847ef561'


def fetch_truecolor_layer(time, bbox, output_dir=None, width=2048):
    print("Starting truecolor...")
    coords_wgs84 = [bbox[0][1], bbox[1][0], bbox[1][1], bbox[0][0]]
    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)

    request = WmsRequest(
        data_folder=output_dir or 'outputs/truecolor',
        layer='TRUE_COLOR',
        bbox=bbox,
        time=time or ('2017-09-01', '2017-10-01'),
        width=width,
        maxcc=0.25,
        instance_id=RIVO_INSTANCE_ID,
    )
    print("Fetching truecolor...")
    request.get_data(save_data=True)


def fetch_s1_layer(name, *, output_dir=None, width=2048, time=None, bbox=None, data_source=DataSource.SENTINEL1_IW):
    print("Starting %s..." % name)
    if bbox is None:
        bbox = COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT

    coords_wgs84 = [bbox[0][1], bbox[1][0], bbox[1][1], bbox[0][0]]
    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)

    data_folder = output_dir or ('outputs/' + name)
    request = WmsRequest(
        data_folder=data_folder,
        data_source=data_source,
        layer=name,
        bbox=bbox,
        time=time or ('2017-09-01', '2017-10-01'),
        width=width,
        instance_id=INSTANCE_ID,
    )
    print("Fetching %s..." % name)
    request.get_data(save_data=True)


def main():
    print("main()")
    return

    # fetch_s1_layer('S1D-VH-VV-VH')
    # fetch_s1_layer('S1D-VV-VV-VH')
    fetch_s1_layer('S1-VV')
    fetch_s1_layer('S1-VH')


if __name__ == '__main__':
    main()
