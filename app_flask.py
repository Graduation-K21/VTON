from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import uuid
import subprocess
import threading
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'

# Hàng đợi công việc đơn giản (sử dụng dictionary để theo dõi trạng thái)
tasks = {}

def run_ai_task(job_id, person_image_path, cloth_image_path):
    """
    Hàm này chạy trong một thread riêng biệt để gọi script AI.
    """
    tasks[job_id] = {'status': 'processing'}
    try:
        # Kích hoạt môi trường ảo và chạy script xử lý AI
        # Chúng ta cần truyền đường dẫn tuyệt đối của python trong venv
        python_executable = os.path.join(os.getcwd(), 'venv', 'bin', 'python')
        script_path = os.path.join(os.getcwd(), 'run_inference_task.py')
        
        command = [
            python_executable,
            script_path,
            '--job_id', job_id,
            '--person_image', person_image_path,
            '--cloth_image', cloth_image_path
        ]
        
        # Chạy command và đợi nó hoàn thành
        subprocess.run(command, check=True)
        
        # Nếu script chạy thành công (không có exception), cập nhật trạng thái
        tasks[job_id]['status'] = 'completed'

    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy tác vụ AI cho job {job_id}: {e}")
        tasks[job_id]['status'] = 'failed'
    except Exception as e:
        print(f"Một lỗi không xác định đã xảy ra với job {job_id}: {e}")
        tasks[job_id]['status'] = 'failed'


@app.route('/')
def index():
    """Phục vụ trang chủ."""
    return render_template('index.html')

@app.route('/tryon')
def tryon_page():
    """Phục vụ trang thử đồ."""
    return render_template('tryon.html')

@app.route('/submit', methods=['POST'])
def submit():
    """Nhận ảnh, tạo job và khởi chạy tác vụ AI."""
    if 'person_image' not in request.files or 'cloth_image' not in request.files:
        return jsonify({'error': 'Thiếu file ảnh.'}), 400

    person_file = request.files['person_image']
    cloth_file = request.files['cloth_image']

    if person_file.filename == '' or cloth_file.filename == '':
        return jsonify({'error': 'Chưa chọn file.'}), 400

    job_id = str(uuid.uuid4())
    job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    os.makedirs(job_upload_dir, exist_ok=True)

    person_filename = secure_filename(person_file.filename)
    cloth_filename = secure_filename(cloth_file.filename)
    
    person_image_path = os.path.join(job_upload_dir, person_filename)
    cloth_image_path = os.path.join(job_upload_dir, cloth_filename)
    
    person_file.save(person_image_path)
    cloth_file.save(cloth_image_path)

    # Chạy tác vụ AI trong một thread nền để không làm block server
    thread = threading.Thread(target=run_ai_task, args=(job_id, person_image_path, cloth_image_path))
    thread.start()

    return jsonify({'job_id': job_id})

@app.route('/status/<job_id>')
def task_status(job_id):
    """Kiểm tra trạng thái của một công việc."""
    result_filename = f"{job_id}.png"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

    if os.path.exists(result_path):
        return jsonify({
            'status': 'completed',
            'result_url': url_for('get_result_file', filename=result_filename)
        })
    
    task = tasks.get(job_id)
    if task:
        return jsonify({'status': task['status']})
    
    return jsonify({'status': 'not_found'}), 404

@app.route('/results/<filename>')
def get_result_file(filename):
    """Phục vụ file ảnh kết quả."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    # Đảm bảo các thư mục tồn tại
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(debug=True, port=5001) 