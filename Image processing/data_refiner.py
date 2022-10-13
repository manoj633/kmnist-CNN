import cv2

# Takes mask image as input, display vertical half of image,
# take clicked values and make a list.
# Then, when pressed 'n', closes the image and displays the next half,
# click on the error causing images and
# finally hit 'q' to end finding errors in that page.
# If it has for loop then you will be required to do the same for multiple images.


# List_name = String_name.split() will split String name into list items
# with space as reference and is stored as List_name.
# String_name.split(',') will split with reference to , (comma)
# String_name = ','.join(List_name) will convert List into string putting ,
# between each element of list

ErrorList = []


def click_event(event, x_clicked, y_clicked, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        ErrorList.append((x_clicked, round(initial_value + y_clicked * 1.5)))


# 1 to 13 // one letter at a time
x = 6
file_pointer = open(r"errors.txt", "a")
txt_list = []

# (1,7) all sheets of a character
for y in range(1, 7):
    image = cv2.imread(
        "G:/Project dataset/Manoj/mine/" + str(x) + str(y) + ".jpg"
    )  # This image is of size x = 1500 by y = 2400
    ErrorList = []

    # ---<DRAWING RECTANGLE TO EACH CHARACTER>---#
    for i in range(1, 16):
        for j in range(1, 25):
            x_val = i * 100 - 100
            y_val = j * 100 - 100
            cv2.rectangle(
                image, (x_val, y_val), (x_val + 100, y_val + 100), (255, 255, 255), 1
            )  # these points should be used to crop the individual charcters.

    mask = image[0 : image.shape[0] // 2, 0 : image.shape[1]]
    # (x,y)=(1500,800), x = mask.shape[1], y = mask.shape[0]
    mask = cv2.resize(mask, (1500, 800))
    initial_value = 0
    cv2.imshow("mask", mask)
    cv2.setMouseCallback("mask", click_event)
    b = 0xFF == ord("n")
    print(b)
    print(type(b))
    cv2.waitKey(b)
    # print(mask.shape[0], mask.shape[1])
    cv2.destroyWindow("mask")

    # Second half of image
    mask2 = image[image.shape[0] // 2 + 1 : image.shape[0], 0 : image.shape[1]]
    mask2 = cv2.resize(mask2, (1500, 800))
    initial_value = 1200
    cv2.imshow("mask2", mask2)
    cv2.setMouseCallback("mask2", click_event)
    cv2.waitKey(0xFF == ord("n"))
    # print(ErrorList)
    # print(mask.shape[0], mask.shape[1])
    list_x_y_with_space = []

    # accessing every element which is a tuple of 2 value, x and y
    for item in ErrorList:
        list_item = list(item)
        # list_item is [x,y] making it ['x','y'], then ['x,y','x,y']
        list_x_y_with_space.append(",".join([str(list_item[0]), str(list_item[1])]))

        # print(list_x_y_with_space)
        Error_items = " ".join(list_x_y_with_space)
        # print(Error_items)
        txt_list.append(str(x) + str(y) + "-" + Error_items)
        cv2.destroyWindow("mask2")

# txt_list has 0 to 5 index (both included) FOR Y=0 TO 6
# print(txt_list)
# counterpart is txt_list = file_pointer.read() ->returns list of strings
# with each string being a line
file_pointer.write("\n" + "\n".join(txt_list))
file_pointer.close()
cv2.destroyAllWindows()
# This code seperates any erroraneous characters from the mnistifiable ones
