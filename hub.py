from sentinelhub import CRS, BBox, WmsRequest, DataSource


# The test site near Võrtsjärv
COORDS_TOP_LEFT, COORDS_BOTTOM_RIGHT = (
    (57.9412725,25.9080495),
    (57.8986439,26.0045503),
)
# Rivo's instance
INSTANCE_ID = '4cb42bbe-04a4-40ad-9c5b-a9a88ba92dc3'


def main():
    print("main()")
    coords_wgs84 = [COORDS_TOP_LEFT[1], COORDS_BOTTOM_RIGHT[0], COORDS_BOTTOM_RIGHT[1], COORDS_TOP_LEFT[0]]
    print(coords_wgs84)
    bbox = BBox(bbox=coords_wgs84, crs=CRS.WGS84)

    print("main() 3")
    # wms_true_color_request = WmsRequest(data_folder='output_dir',
    #                                     layer='TRUE_COLOR',
    #                                     bbox=bbox,
    #                                     #time='2017-12-15',
    #                                     time=('2017-06-01', '2017-06-30'),
    #                                     width=2048,
    #                                     maxcc=0.2,
    #                                     #height=856,
    #                                     #instance_id=INSTANCE_ID
    #                                     )
    wms_true_color_request = WmsRequest(
        data_folder='output_dir',
        data_source=DataSource.SENTINEL1_IW,
        # layer='S1-VH',
        layer='NEW-FALSE-COLOR-VEGETATION',
        bbox=bbox,
        time=('2016-04-01', '2017-07-01'),
        width=2048,
        # maxcc=0.5,
        instance_id=INSTANCE_ID,
    )
    # wms_true_color_request = WmsRequest(
    #     data_folder='output_dir',
    #     data_source=DataSource.SENTINEL2_L1C,
    #     layer='NEW-FALSE-COLOR-VEGETATION',
    #     bbox=bbox,
    #     time=('2017-05-01', '2017-06-30'),
    #     width=2048,
    #     maxcc=0.2,
    #     instance_id=INSTANCE_ID,
    # )
    #
    print("main() 4")
    wms_true_color_img = wms_true_color_request.get_data(save_data=True)

    print('Returned data is of type = %s and length %d.' % (type(wms_true_color_img), len(wms_true_color_img)))
    print('Single element in the list is of type = {} and has shape {}'.format(type(wms_true_color_img[-1]),
                                                                               wms_true_color_img[-1].shape))


if __name__ == '__main__':
    main()
