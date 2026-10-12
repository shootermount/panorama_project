import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time
from collections import deque


MIN_IMAGES = 2
MAX_IMAGES = 10

MAX_IMAGE_SIZE = 1400        
RATIO_TEST = 0.75            
RANSAC_THRESHOLD = 3.0       
MIN_INLIERS = 15             
MIN_INLIER_RATIO = 0.25      

MAX_CANVAS_SIDE = 6000       
MAX_CANVAS_PIXELS = 25_000_000


CROP_TO_RECTANGLE = True


RESULTS_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)



def load_image(path):

    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    height, width = image.shape[:2]

    scale = min(max_size / max(height, width), 1.0)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    return image



def prepare_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def extract_features(image):
 
    gray = prepare_image(image)

    sift = cv2.SIFT_create(
        nfeatures=4000,
        contrastThreshold=0.02
    )

    return sift.detectAndCompute(gray, None)


def match_pair(features_i, features_j):
    """
    Возвращает гомографию H, которая переводит точки
    фото j в систему координат фото i.
    """

    keypoints_i, descriptors_i = features_i
    keypoints_j, descriptors_j = features_j

    if descriptors_i is None or descriptors_j is None:
        return None

    if len(keypoints_i) < 8 or len(keypoints_j) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    raw_matches = matcher.knnMatch(
        descriptors_j,
        descriptors_i,
        k=2
    )

    
    good_matches = []

    for pair in raw_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    if len(good_matches) < 8:
        return None

    src_points = np.float32([
        keypoints_j[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        keypoints_i[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_points,
        dst_points,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
        maxIters=5000,
        confidence=0.995
    )

    if H is None or mask is None:
        return None

    if abs(H[2, 2]) < 1e-12:
        return None

    H = H / H[2, 2]

    return {
        "H": H,
        "matches": good_matches,
        "mask": mask.ravel().astype(bool),
        "inliers": int(mask.sum()),
        "n_matches": len(good_matches),
    }



def check_homography(H, image_shape):

    if H is None or not np.isfinite(H).all():
        return False

    height, width = image_shape[:2]

    corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    denominator = (
        H[2, 0] * corners[:, 0]
        + H[2, 1] * corners[:, 1]
        + H[2, 2]
    )

    if np.any(denominator <= 1e-6):
        return False

    warped = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)


    for k in range(4):
        a = warped[k]
        b = warped[(k + 1) % 4]
        c = warped[(k + 2) % 4]

        cross = (
            (b[0] - a[0]) * (c[1] - b[1])
            - (b[1] - a[1]) * (c[0] - b[0])
        )

        if cross <= 0:
            return False


    x = warped[:, 0]
    y = warped[:, 1]

    area = 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )

    ratio = area / (width * height)

    return 0.2 < ratio < 5.0



def build_match_graph(images, features, names):
    """
    Сопоставляет ВСЕ пары фото. Поэтому порядок, в котором
    пользователь выбрал файлы, больше не важен.
    """

    edges = {}

    for i in range(len(images)):
        for j in range(i + 1, len(images)):

            result = match_pair(features[i], features[j])

            if result is None:
                print(f"{names[i]} <-> {names[j]}: нет связи")
                continue

            ratio = result["inliers"] / result["n_matches"]

            valid = (
                result["inliers"] >= MIN_INLIERS
                and ratio >= MIN_INLIER_RATIO
                and check_homography(result["H"], images[j].shape)
            )

            print(
                f"{names[i]} <-> {names[j]}: "
                f"совпадений {result['n_matches']}, "
                f"инлайеров {result['inliers']} "
                f"({ratio:.0%}) -> "
                f"{'OK' if valid else 'отброшено'}"
            )

            if valid:
                edges[(i, j)] = result

    return edges


def build_spanning_tree(count, edges):
    """
    Максимальное остовное дерево: оставляем только самые
    надёжные связи (с наибольшим числом инлайеров).
    """

    parent = list(range(count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = []

    ordered_edges = sorted(
        edges.items(),
        key=lambda item: item[1]["inliers"],
        reverse=True
    )

    for (i, j), _ in ordered_edges:

        root_i = find(i)
        root_j = find(j)

        if root_i != root_j:
            parent[root_i] = root_j
            tree.append((i, j))

    return tree


def bfs(adjacency, start):
    parent = {start: None}
    distance = {start: 0}
    order = [start]

    queue = deque([start])

    while queue:
        u = queue.popleft()

        for v in adjacency[u]:
            if v not in distance:
                distance[v] = distance[u] + 1
                parent[v] = u
                order.append(v)
                queue.append(v)

    return order, parent, distance


def compute_global_transforms(count, edges, tree):
    """
    Выбирает опорное фото (центр цепочки) и считает матрицу
    для каждого фото относительно него.
    Центральное опорное фото сильно уменьшает искажения по краям.
    """

    adjacency = {k: [] for k in range(count)}

    for i, j in tree:
        adjacency[i].append(j)
        adjacency[j].append(i)

    
    visited = set()
    best_component = []

    for start in range(count):

        if start in visited:
            continue

        order, _, _ = bfs(adjacency, start)
        visited.update(order)

        if len(order) > len(best_component):
            best_component = order

    reference = None
    best_eccentricity = None

    for node in best_component:

        _, _, distance = bfs(adjacency, node)
        eccentricity = max(distance.values())

        if best_eccentricity is None or eccentricity < best_eccentricity:
            best_eccentricity = eccentricity
            reference = node

    order, parent, _ = bfs(adjacency, reference)

    transforms = {reference: np.eye(3)}

    for node in order[1:]:

        p = parent[node]


        if (p, node) in edges:
            H_node_to_p = edges[(p, node)]["H"]
        else:
            H_node_to_p = np.linalg.inv(edges[(node, p)]["H"])

        H_total = transforms[p] @ H_node_to_p
        transforms[node] = H_total / H_total[2, 2]

    return reference, transforms


def largest_inscribed_rectangle(mask):
    """
    Ищет самый большой прямоугольник, целиком лежащий внутри
    области mask (True = есть изображение).
    Возвращает (x, y, ширина, высота).
    """

    height, width = mask.shape

    heights = [0] * width

    best_area = 0
    best_rect = (0, 0, 0, 0)

    for row in range(height):

        row_values = mask[row].tolist()

        for col in range(width):
            heights[col] = heights[col] + 1 if row_values[col] else 0


        stack = []

        for col in range(width + 1):

            current = heights[col] if col < width else 0
            start = col

            while stack and stack[-1][1] >= current:

                start, top_height = stack.pop()
                area = top_height * (col - start)

                if area > best_area:
                    best_area = area
                    best_rect = (
                        start,
                        row - top_height + 1,
                        col - start,
                        top_height
                    )

            stack.append((start, current))

    return best_rect


def crop_to_rectangle(panorama, valid):
    """
    Обрезает панораму до ровного прямоугольника без чёрных полей.
    """

    height, width = valid.shape

    scale = min(1.0, 500 / max(height, width))

    small = cv2.resize(
        valid.astype(np.uint8) * 255,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA
    ) >= 255

    x, y, w, h = largest_inscribed_rectangle(small)

    if w == 0 or h == 0:
        return None

    x0 = int(np.ceil(x / scale)) + 2
    y0 = int(np.ceil(y / scale)) + 2
    x1 = int(np.floor((x + w) / scale)) - 2
    y1 = int(np.floor((y + h) / scale)) - 2

    for _ in range(30):

        if x1 <= x0 or y1 <= y0:
            return None

        if valid[y0:y1, x0:x1].all():
            return panorama[y0:y1, x0:x1]

        x0 += 1
        y0 += 1
        x1 -= 1
        y1 -= 1

    return None


def render_panorama(images, transforms):

    ids = sorted(transforms)

    all_corners = []

    for k in ids:
        h, w = images[k].shape[:2]

        corners = np.float32([
            [0, 0], [w, 0], [w, h], [0, h]
        ]).reshape(-1, 1, 2)

        all_corners.append(
            cv2.perspectiveTransform(
                corners,
                transforms[k]
            ).reshape(-1, 2)
        )

    all_corners = np.concatenate(all_corners)

    xmin, ymin = np.floor(all_corners.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_corners.max(axis=0)).astype(int)

    width = int(xmax - xmin)
    height = int(ymax - ymin)

    if width <= 0 or height <= 0 or max(width, height) > 20 * MAX_IMAGE_SIZE:
        raise ValueError(
            "Панорама получилась неадекватного размера. "
            "Совпадения, вероятно, определены неправильно."
        )


    scale = min(
        1.0,
        MAX_CANVAS_SIDE / max(width, height),
        (MAX_CANVAS_PIXELS / (width * height)) ** 0.5
    )

    base = np.array([
        [scale, 0, -xmin * scale],
        [0, scale, -ymin * scale],
        [0, 0, 1]
    ], dtype=np.float64)

    canvas_width = int(np.ceil(width * scale))
    canvas_height = int(np.ceil(height * scale))

    accumulator = np.zeros(
        (canvas_height, canvas_width, 3),
        dtype=np.float32
    )

    weight_sum = np.zeros(
        (canvas_height, canvas_width),
        dtype=np.float32
    )

    kernel = np.ones((3, 3), np.uint8)

    for k in ids:

        M = base @ transforms[k]

        warped = cv2.warpPerspective(
            images[k],
            M,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR
        )


        mask = cv2.warpPerspective(
            np.full(images[k].shape[:2], 255, np.uint8),
            M,
            (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST
        )

        mask = cv2.erode(mask, kernel)


        weight = cv2.distanceTransform(
            mask,
            cv2.DIST_L2,
            5
        )

        accumulator += warped.astype(np.float32) * weight[..., None]
        weight_sum += weight

    valid = weight_sum > 0

    panorama = np.zeros_like(accumulator)
    panorama[valid] = accumulator[valid] / weight_sum[valid][:, None]
    panorama = np.uint8(np.clip(panorama, 0, 255))

    if CROP_TO_RECTANGLE:

        cropped = crop_to_rectangle(panorama, valid)

        if cropped is not None:

            kept = cropped.shape[0] * cropped.shape[1] / valid.sum()

    
            if kept >= 0.10:
                print(f"Обрезка до прямоугольника: оставлено {kept:.0%} панорамы")
                return cropped

        print("Не удалось обрезать до прямоугольника, обрезаю по краям")

    points = cv2.findNonZero(valid.astype(np.uint8))

    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        panorama = panorama[y:y + h, x:x + w]

    return panorama



def build_panorama(images, names, results_folder=RESULTS_FOLDER):

    os.makedirs(results_folder, exist_ok=True)

    print("Поиск признаков...")
    features = [extract_features(image) for image in images]

    for name, (keypoints, _) in zip(names, features):
        print(f"{name}: {len(keypoints)} ключевых точек")

    print()
    print("Сопоставление всех пар...")
    edges = build_match_graph(images, features, names)

    if not edges:
        raise ValueError(
            "Ни одна пара фотографий не совпала.\n"
            "Проверьте, что фото сняты из одной точки "
            "и имеют 30-50% общего изображения."
        )

    tree = build_spanning_tree(len(images), edges)

    reference, transforms = compute_global_transforms(
        len(images),
        edges,
        tree
    )

    used = sorted(transforms)
    skipped = [k for k in range(len(images)) if k not in transforms]

    if len(used) < 2:
        raise ValueError("Не удалось соединить фотографии.")

    print()
    print(f"Опорное (центральное) фото: {names[reference]}")

    total_matches = 0
    total_inliers = 0

    for i, j in tree:

        if i not in transforms or j not in transforms:
            continue

        result = edges[(i, j)]

        total_matches += result["n_matches"]
        total_inliers += result["inliers"]

        visualization = cv2.drawMatches(
            images[j],
            features[j][0],
            images[i],
            features[i][0],
            result["matches"],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=result["mask"].astype(int).tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        cv2.imwrite(
            os.path.join(results_folder, f"matches_{i + 1}_{j + 1}.jpg"),
            visualization
        )

    print("Сборка панорамы...")
    panorama = render_panorama(images, transforms)

    info = {
        "used": used,
        "skipped": skipped,
        "reference": reference,
        "pairs": len(tree),
        "matches": total_matches,
        "inliers": total_inliers,
    }

    return panorama, info




def show_panorama(panorama):

    height, width = panorama.shape[:2]

    scale = min(1.0, 1400 / width, 800 / height)

    view = panorama

    if scale < 1:
        view = cv2.resize(
            panorama,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA
        )

    title = "Готовая панорама (любая клавиша - закрыть)"

    cv2.imshow(title, view)


    while True:
        key = cv2.waitKey(100)

        if key != -1:
            break

        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()



def select_images():

    file_paths = filedialog.askopenfilenames(
        title="Выберите фотографии для панорамы",
        filetypes=[
            (
                "Изображения",
                "*.jpg *.jpeg *.png *.bmp"
            )
        ]
    )

    if not file_paths:
        return

    if len(file_paths) < MIN_IMAGES:
        messagebox.showerror(
            "Ошибка",
            f"Выберите минимум {MIN_IMAGES} фотографии."
        )
        return

    if len(file_paths) > MAX_IMAGES:
        messagebox.showerror(
            "Ошибка",
            f"Можно выбрать максимум {MAX_IMAGES} фотографий."
        )
        return

    start_time = time.time()

    button.config(state="disabled")
    status_label.config(text="Обработка... подождите")
    root.update()

    try:
    
        images = []
        names = []

        for path in file_paths:

            image = load_image(path)

            if image is None:
                messagebox.showerror(
                    "Ошибка",
                    f"Не удалось открыть:\n{path}"
                )
                return

            images.append(resize_image(image))
            names.append(os.path.basename(path))

        try:
            panorama, info = build_panorama(images, names)

        except ValueError as error:
            messagebox.showerror(
                "Ошибка при создании панорамы",
                f"{error}\n\n"
                f"Совет: фотографии должны иметь "
                f"30-50% общего изображения."
            )
            return

        
        output_path = os.path.join(
            RESULTS_FOLDER,
            "panorama_final.jpg"
        )

        cv2.imwrite(output_path, panorama)

        
        processing_time = time.time() - start_time
        height, width = panorama.shape[:2]

        inlier_ratio = 0

        if info["matches"] > 0:
            inlier_ratio = info["inliers"] / info["matches"]

        statistics = (
            f"Выбрано фотографий: {len(images)}\n"
            f"Использовано в панораме: {len(info['used'])}\n"
            f"Опорное фото: {names[info['reference']]}\n"
            f"Использовано связей: {info['pairs']}\n"
            f"Хороших совпадений: {info['matches']}\n"
            f"RANSAC-инлайеров: {info['inliers']}\n"
            f"Доля инлайеров: {inlier_ratio:.2%}\n"
            f"Размер панорамы: {width} x {height} px\n"
            f"Время обработки: {processing_time:.2f} сек."
        )

        if info["skipped"]:
            skipped_names = ", ".join(names[k] for k in info["skipped"])
            statistics += f"\n\nНЕ вошли в панораму: {skipped_names}"

        print()
        print("==========================================")
        print("ПАНОРАМА ГОТОВА")
        print("==========================================")
        print(statistics)
        print()
        print(f"Файл: {output_path}")

        show_panorama(panorama)

        messagebox.showinfo(
            "Готово",
            f"Панорама создана!\n\n"
            f"{statistics}\n\n"
            f"Сохранено:\n{output_path}"
        )

    finally:
        button.config(state="normal")
        status_label.config(text="")



if __name__ == "__main__":

    root = tk.Tk()

    root.title("Создание панорамы — Computer Vision")
    root.geometry("500x340")

    title_label = tk.Label(
        root,
        text="Создание панорамы",
        font=("Arial", 20, "bold")
    )
    title_label.pack(pady=25)

    info_label = tk.Label(
        root,
        text=(
            f"Выберите от {MIN_IMAGES} до {MAX_IMAGES} фотографий.\n"
            "Порядок выбора не важен.\n\n"
            "Рекомендуется, чтобы соседние фотографии\n"
            "имели 30–50% общего изображения."
        ),
        font=("Arial", 11),
        justify="center"
    )
    info_label.pack(pady=10)

    methods_label = tk.Label(
        root,
        text=(
            "SIFT • BFMatcher • Ratio Test\n"
            "RANSAC • Homography • Feather Blending"
        ),
        font=("Arial", 10)
    )
    methods_label.pack(pady=10)

    button = tk.Button(
        root,
        text="Выбрать фотографии",
        command=select_images,
        font=("Arial", 12),
        width=25,
        height=2
    )
    button.pack(pady=15)

    status_label = tk.Label(
        root,
        text="",
        font=("Arial", 10),
        fg="gray"
    )
    status_label.pack()

    root.mainloop()
