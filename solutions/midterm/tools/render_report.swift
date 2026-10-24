import Foundation
import CoreGraphics
import CoreText
import PDFKit
import AppKit
let args=CommandLine.arguments
if args.count > 1 && args[1] == "--extract" {
 let source=PDFDocument(url:URL(fileURLWithPath:args[2]))!
 let text=(0..<source.pageCount).map {source.page(at:$0)?.string ?? ""}.joined(separator:"\n")
 try text.write(toFile:args[3],atomically:true,encoding:.utf8)
 print("Extracted \(source.pageCount) pages")
 exit(0)
}
let blocks = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath:args[1]))) as! [[String:String]]
let output=URL(fileURLWithPath:args[2]); var box=CGRect(x:0,y:0,width:595,height:842)
guard let ctx=CGContext(output as CFURL,mediaBox:&box,nil) else {fatalError("PDF context failed")}
var y:CGFloat=0, pages=0
func draw(_ s:String,_ f:String,_ size:CGFloat,_ x:CGFloat,_ y:CGFloat){let a=NSAttributedString(string:s,attributes:[.font:NSFont(name:f,size:size)!, .foregroundColor:NSColor(calibratedWhite:0.25,alpha:1)]);ctx.textPosition=CGPoint(x:x,y:y);CTLineDraw(CTLineCreateWithAttributedString(a),ctx)}
func endPage(){draw("Gleb Alektorov  |  Recsys & LLMs  |  Midterm", "Helvetica",8,48,29);draw(String(pages),"Helvetica",8,535,29);ctx.endPDFPage()}
func newPage(){if pages>0 {endPage()};ctx.beginPDFPage(nil);pages += 1;y=790;ctx.setFillColor(CGColor(red:0.10,green:0.27,blue:0.40,alpha:1));ctx.fill(CGRect(x:48,y:805,width:499,height:3))}
newPage()
for b in blocks {
 let kind=b["kind"] ?? "body", text=b["text"] ?? ""
 if kind=="page" {newPage();continue}
 if kind=="image" {
  let ns=NSImage(contentsOfFile:text)!
  var rect=CGRect(origin:.zero,size:ns.size)
  let im=ns.cgImage(forProposedRect:&rect,context:nil,hints:nil)!
  let height=499 * CGFloat(im.height)/CGFloat(im.width)
  if y-height<52 {newPage()}
  ctx.draw(im,in:CGRect(x:48,y:y-height,width:499,height:height));y-=height+12
  continue
 }
 let size:CGFloat=kind=="title" ? 24 : kind=="h1" ? 15 : kind=="h2" ? 11 : kind=="small" ? 8.3 : 9.5
 let font=CTFontCreateWithName((kind=="title" || kind=="h1" || kind=="h2" ? "Helvetica-Bold" : "Helvetica") as CFString,size,nil)
 let attr=CFAttributedStringCreate(nil,text as CFString,[kCTFontAttributeName:font,kCTForegroundColorAttributeName:CGColor(gray:0.12,alpha:1)] as CFDictionary)!
 let ts=CTTypesetterCreateWithAttributedString(attr);var pos=0;let n=(text as NSString).length
 if (kind=="h1" || kind=="h2") && y<155 {newPage()}
 while pos<n {let count=max(1,CTTypesetterSuggestLineBreak(ts,pos,499));let line=CTTypesetterCreateLine(ts,CFRange(location:pos,length:min(count,n-pos)));var asc:CGFloat=0,desc:CGFloat=0,lead:CGFloat=0;_=CTLineGetTypographicBounds(line,&asc,&desc,&lead);let h=max(size*1.35,asc+desc+lead+2)
 if y-h<52 {newPage()};ctx.textPosition=CGPoint(x:48,y:y-asc);CTLineDraw(line,ctx);y-=h;pos+=count
 }
 y -= kind=="h1" ? 10 : kind=="small" ? 3 : 7
}
endPage();ctx.closePDF()
let pdf=PDFDocument(url:output)!
let link="https://github.com/Moonkato/llm4rec_midterm/tree/main/solutions/midterm"
for selection in pdf.findString(link,withOptions:[]) {for pg in selection.pages {let ann=PDFAnnotation(bounds:selection.bounds(for:pg),forType:.link,withProperties:nil);ann.url=URL(string:link);pg.addAnnotation(ann)}}
pdf.write(to:output)
var extracted=""
for i in 0..<pdf.pageCount {
 let page=pdf.page(at:i)!;extracted += (page.string ?? "")+"\n"
 let bounds=page.bounds(for:.mediaBox), scale:CGFloat=1.2
 let w=Int(bounds.width*scale),h=Int(bounds.height*scale)
 let c=CGContext(data:nil,width:w,height:h,bitsPerComponent:8,bytesPerRow:0,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue)!
 c.setFillColor(CGColor(gray:1,alpha:1));c.fill(CGRect(x:0,y:0,width:w,height:h));c.scaleBy(x:scale,y:scale);page.draw(with:.mediaBox,to:c)
 let rep=NSBitmapImageRep(cgImage:c.makeImage()!);try rep.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:args[3]+String(format:"/page-%02d.png",i+1)))
}
try extracted.write(toFile:args[3]+"/extracted.txt",atomically:true,encoding:.utf8)
print("PDF pages: \(pdf.pageCount); extracted characters: \(extracted.count)")
