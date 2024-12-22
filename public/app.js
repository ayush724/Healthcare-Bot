var express=require("express"); 
var bodyParser=require("body-parser"); 

const mongoose = require('mongoose');
// mongo password 67rnNTG9vfBNziLx 
mongoose.connect('mongodb+srv://ayush:67rnNTG9vfBNziLx@cluster0.ie4iaae.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'); 
var db=mongoose.connection; 
db.on('error', console.log.bind(console, "connection error")); 
db.once('open', function(callback){ 
	console.log("connection succeeded"); 
}) 

var app=express() 


app.use(bodyParser.json()); 
app.use(express.static('public')); 
app.use(bodyParser.urlencoded({ 
	extended: true
})); 

app.post('/login', function(req,res){ 
	var name = req.body.name; 
	var email =req.body.email; 
	var pass = req.body.password;  

	var data = { 
		"name": name, 
		"email":email, 
		"password":pass, 
	} 
db.collection('details').insertOne(data,function(err, collection){ 
		if (err) throw err; 
		console.log("Record inserted Successfully"); 
			
	}); 
		
	return res.redirect('./frontpage2.html'); 
}) 


app.get('/',function(req,res){ 
res.set({ 
	'Access-control-Allow-Origin': '*'
	}); 
return res.redirect('frontpage2.html'); 
}).listen(3000) 


console.log("server listening at port 3000");
