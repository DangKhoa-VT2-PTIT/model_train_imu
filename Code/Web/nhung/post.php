<?php
$tempF = $_POST["temp"];
$file = 'temp.html';
$data = $tempF;
file_put_contents($file, $data);
echo $data;
?>
