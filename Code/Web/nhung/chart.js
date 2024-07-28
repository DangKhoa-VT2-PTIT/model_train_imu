// Lấy phần tử canvas từ HTML
const ctx = document.getElementById('myChart').getContext('2d');

// Dữ liệu ban đầu và cấu hình của biểu đồ
let data = {
    labels: ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'],
    datasets: [{
        label: 'Walking',
        data: [-0.0823, 0.992584, -0.16158, 2.349982, 23.98565, -2.51637],
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1
    },
    {
        label: 'Standing',
        data: [0.101094, 0.988854, -0.08905, -3.27173, 1.309229, -0.55578],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1
    },
    {
        label: 'Sitting',
        data: [-0.95471, 0.15476, 0.12724, -3.26344, 1.29425, -0.53861],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
    },
    {
        label: 'Running',
        data: [-0.35118, 0.821981, -0.29795, -12.14, 9.6375, 0.037263],
        backgroundColor: 'rgba(255, 159, 64, 0.2)',
        borderColor: 'rgba(255, 159, 64, 1)',
        borderWidth: 1
    },
    {
        label: 'Jumping',
        data: [-0.40693, 0.809674, -0.30839, 0.089899, 1.715159, -0.27371],
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        borderColor: 'rgba(153, 102, 255, 1)',
        borderWidth: 1
    },]
};

const config = {
    type: 'line',
    data: data,
    options: {
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
};

// Hàm để thêm giá trị ngẫu nhiên cho nhãn "Test"
function addRandomData() {
    // Khởi tạo mảng chứa 6 giá trị ngẫu nhiên từ 1 đến 20
    const randomData = Array.from({ length: 6 }, () => Math.floor(Math.random() * 20) + 1);
    
    data.datasets.push({
        label: 'Data',
        // data: randomData,
        data: [-0.93,	0.16,	0.13,	-3.16,	1.66,	-0.6

        ],

        // data: [-0.40693, 0.809674, -0.30839, 0.089899, 1.715159, -0.27371],
        backgroundColor: 'rgba(255, 206, 86, 0.2)',
        borderColor: 'rgba(255, 206, 86, 1)',
        borderWidth: 1
    });
    
    // Cập nhật biểu đồ
    myChart.update();
}

// Tạo biểu đồ mới
const myChart = new Chart(ctx, config);

// Gọi hàm thêm giá trị ngẫu nhiên khi trang được load
addRandomData();









//////FFFFFFFFFFFFFFFFF

// Hàm để thêm giá trị ngẫu nhiên cho nhãn "Test"
// function addRandomData() {
//     // Khởi tạo mảng chứa 6 giá trị ngẫu nhiên từ 1 đến 20
//     const randomData = Array.from({ length: 6 }, () => Math.floor(Math.random() * 20) + 1);
    
//     data.datasets.push({
//         label: 'Data',
//         data: randomData,
//         isHighlight: true, // Đánh dấu là đường 'Data'
//         backgroundColor: 'rgba(255, 206, 86, 0.2)',
//         borderColor: 'rgba(255, 206, 86, 1)',
//         borderWidth: 1
//     });
    
//     // Cập nhật biểu đồ
//     updateChart("Standing");
// }


// // Hàm để cập nhật biểu đồ
// function updateChart(receivedData) {
//     // Khởi tạo mảng chứa các nhãn được xác định và 'Data'
//     let filteredDatasets = [];

//     // Chọn dataset tương ứng với dữ liệu nhận được
//     switch (receivedData) {
//         case 'Running':
//         case 'Walking':
//         case 'Sitting':
//         case 'Standing':
//         case 'Jumping':
//             filteredDatasets = data.datasets.filter(dataset => dataset.label === receivedData || dataset.label === 'Data');
//             break;
//         default:
//             // Trường hợp mặc định nếu nhãn không khớp với các trường hợp trên
//             filteredDatasets = data.datasets.filter(dataset => dataset.label === 'Data');
//             break;
//     }
    
//     // Cập nhật cấu hình biểu đồ
//     myChart.config.data.datasets = filteredDatasets;

//     // Đặt độ mờ cho các đường không phải 'Running', 'Walking', 'Sitting', 'Standing', 'Jumping' hoặc 'Data'
//     myChart.config.data.datasets.forEach(dataset => {
//         if (!['Running', 'Walking', 'Sitting', 'Standing', 'Jumping', 'Data'].includes(dataset.label)) {
//             dataset.borderColor = dataset.borderColor.replace('1)', '0.2)');
//             dataset.backgroundColor = dataset.backgroundColor.replace('1)', '0.2)');
//         }
//     });

//     // Cập nhật biểu đồ
//     myChart.update();
// }



// // Tạo biểu đồ mới
// const myChart = new Chart(ctx, config);

// // Gọi hàm thêm giá trị ngẫu nhiên khi trang được load
// addRandomData();


