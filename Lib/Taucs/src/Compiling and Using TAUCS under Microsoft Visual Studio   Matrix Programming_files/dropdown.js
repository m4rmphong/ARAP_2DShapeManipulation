jQuery(document).ready(function($){
	$("#navmenu a").removeAttr('title');
	$("#navmenu ul").css({display: "none"});
	
	//size and position of master container
	var jWrapper = $("#wrapper");
	var p = jWrapper.offset();
	var w = jWrapper.width();

	$("#navmenu li").hover(
		function(){
			var jSublist = $('ul:first',this);
			jSublist.css({visibility: "hidden",display: "block"});
			var pS = jSublist.offset();
			var wS = jSublist.width();
			jSublist.css({display: "none"});
			if ((pS.left + wS) > (p.left + w)){
				jSublist.css({left:"-145px"});
			}
			$(this).find('ul:first').css({visibility: "visible",display: "none"}).show(250);
		},
		function(){
			$(this).find('ul:first').css({visibility: "hidden"});
		}
	);
	/*
	$("#navmenu li").hover(function(){
		$(this).find('ul:first').css({visibility: "visible",display: "none"}).show(250);
		},function(){
		$(this).find('ul:first').css({visibility: "hidden"});
		});
	*/

	
	//prevent default link if Categories link is clicked
	$("#cat-on-menu a[name=cat-on-menu]").click(function(e){ e.preventDefault(); });
		
});